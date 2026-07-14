# Recipe: Dual-card 27B with MTP speculative decode on vLLM

> **Result (measured):** A 27B model tensor-paralleled across two RTX 3090s, with the checkpoint's
> **MTP (multi-token-prediction) head** driving speculative decode at n=3 and an FP8 (e4m3) KV
> cache, decodes at **79 → 60 t/s from 0 → 146K context fill** — about **2.6–2.7× the no-MTP
> baseline at every depth**, and the curve stays flat as context grows. Plus the one trap that
> will OOM you mid-prefill if you don't know it.

## What this is for

Getting usable single-user decode speed out of a 27B on 24 GB cards means two 3090s in
tensor-parallel — but TP alone doesn't make decode fast, it makes the model *fit* with room for
long context. The decode lever on these checkpoints is the **MTP head** they ship: point vLLM's
speculative decoder at it and you get a large, roughly depth-independent decode multiplier. The
catch is a memory-profiling gap that makes a util value that boots fine die 60K tokens into a
prompt.

## Hardware

| Component | Configuration |
|---|---|
| GPU | 2× RTX 3090 24 GB, Ampere, tensor-parallel (measured on a 3090 Ti + a 3090 — both 24 GB, decode-neutral) |
| GPU topology | one x16 + one x4 — decode-neutral at TP=2 (see note) |
| CPU | AMD Ryzen 9 9900X |
| OS / driver | Linux, NVIDIA 5xx, CUDA 12.x |

## Model and runtime

This recipe's launch block and all depth-sweep numbers are the **INT8 (AutoRound)** build — it's
portable (no engine patches) and it's what the measured curve below came from. A parallel **NVFP4**
build of the same 27B is noted as a variant where it differs.

- Model: a Qwen3.6-27B checkpoint that **ships an MTP head** (vLLM auto-discovers it from the
  model directory — e.g. a `model_extra_tensors.safetensors` alongside the weights).
- Runtime: **vLLM 0.22.1**.
- Context: 151,552 per slot.
- Concurrency: the INT8 pool is sized **single-stream** (`--max-num-seqs 1`, context-over-concurrency);
  the lighter NVFP4 build fit **3× 151K** concurrent slots in the same pool.
- KV cache: FP8 (e4m3).
- `kv_heads = 4` on this checkpoint, so **TP must divide 4** — TP=2 or TP=4 only, never 3.

## Launch

INT8 (AutoRound) build — the measured config:

```bash
export VLLM="/path/to/venv/bin/vllm"
export MODEL_PATH="/path/to/qwen3.6-27b-int8"     # dir containing weights + the MTP head tensors
export PATH="$(dirname "$VLLM"):$PATH"            # ninja on PATH for inductor JIT

export CUDA_VISIBLE_DEVICES=0,1                    # the two 3090s
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASHINFER           # the e4m3-KV decode path — load-bearing (see below)
export VLLM_USE_FLASHINFER_SAMPLER=1
export NCCL_P2P_DISABLE=1                          # no NVLink; disable P2P probing

"$VLLM" serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 8000 \
  --quantization auto_round --dtype float16 \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len 151552 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code
```

**NVFP4 variant of the same 27B** — differs in four places: (1) drop `--kv-cache-dtype fp8` (an NVFP4
checkpoint declares FP8 KV and vLLM auto-selects e4m3); (2) use `--quantization modelopt` instead of
`auto_round`; (3) **also drop `VLLM_ATTENTION_BACKEND=FLASHINFER`** — the FlashInfer choice above is
specific to the INT8-KV decode path and was not benched on NVFP4, so leave the backend at default
(keep `VLLM_USE_FLASHINFER_SAMPLER=1`); (4) it fits `--gpu-memory-utilization 0.90 --max-num-seqs 3`.
Note that some NVFP4 builds that quantize `lm_head` needed extra vLLM patches on versions < 0.24.0 —
prefer ≥ 0.24.0 (see the [Ampere FP8/NVFP4 recipe](ampere-fp8-nvfp4-marlin.md)).

> The reference INT8 launcher also passes `--compilation-config
> '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'` — omitted above for
> portability, but it was part of the exact build that produced the measured 79→60 curve.

## Why these settings

- **`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`** — the whole point. Uses
  the checkpoint's own MTP head to propose 3 tokens per step; vLLM verifies them in one forward
  pass. This is the ~2.6× decode lever. Without it, TP-pair decode on this checkpoint is
  indistinguishable from a plain baseline. n=3 was the sweet spot in testing.
- **`--mamba-cache-mode align`** — required alongside MTP for this architecture's cache layout.
  Stable with prefix caching on.
- **`--kv-cache-dtype fp8` (e4m3) + `VLLM_ATTENTION_BACKEND=FLASHINFER`** — a pair, not two separate
  choices. FP8 KV halves the KV footprint so 151K context is affordable; the FlashInfer backend is
  what keeps e4m3-KV decode **flat with depth** (the 79→60 curve). Without FlashInfer this
  checkpoint's KV decode collapses as context grows. TTFT tax ~+10% (not the 2× some reports claim).
  **Not** `fp8_e5m2` against an FP8-quantized checkpoint.
- **`--gpu-memory-utilization 0.93`** — the ceiling. **The trap, see below** — with MTP on 24 GB
  cards, do **not** exceed 0.93.
- **`--quantization auto_round --dtype float16`** — for the INT8 AutoRound build. (The NVFP4 variant
  uses `--quantization modelopt` and omits the KV-dtype flag; see the variant note above.)
- **`--tensor-parallel-size 2 --disable-custom-all-reduce`** — TP=2 fits the model with long-context
  headroom. Disable the custom all-reduce on a no-NVLink pair (paired with `NCCL_P2P_DISABLE=1`).
- **`--max-num-seqs 1`** — context-over-concurrency: the INT8 pool is spent entirely on one deep
  151K stream. How many 151K slots a build can hold is weight-dependent — the lighter NVFP4 build fit 3.
- **Tradeoff:** MTP's speedup is real but **acceptance is workload-dependent** — see results.

## ⚠ The trap: MTP's activation peak is invisible to startup profiling

vLLM sizes the KV pool at startup by profiling peak memory — **but that profile does not include
the MTP drafter's activation peak.** So:

- `--gpu-memory-utilization 0.95` **launches fine**, serves short prompts fine…
- …then **dies mid-prefill at ~60K fill** — once as a clean 68 MiB OOM, once as a
  `CUDA illegal memory access` (which *looks* like an align-mode/prefix-cache kernel bug but is
  just memory pressure).

**Fix: cap util at ≤ 0.93 when MTP is on, on 24 GB cards.** Prefix caching + `align` mode is stable
there. If you see illegal-memory-access deep into a long prompt, lower util before you go
kernel-hunting.

## Validation

```bash
# Coherence + a decode read
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"Explain MTP speculative decoding in 3 sentences."}],"max_tokens":-1}'
```

Then prove it survives depth — send a ~100K-token prompt with a fact planted near the end and
confirm the model both **retrieves it** (needle-in-a-haystack) and **doesn't OOM** during the long
prefill. A config that passes a short prompt but fails here is the util trap.

## Results

Qwen3.6-27B (INT8 AutoRound build), 2× RTX 3090 TP=2, FP8 e4m3 KV, MTP n=3, util 0.93, ctx 151552.
Measured 2026-07-09 (`depth-sweep.py`, decode-at-depth + TTFT + NIAH, one JSONL row per request):

| Metric | Value | Conditions |
|---|---|---|
| Decode, 0K fill | 79 t/s | single stream |
| Decode, 146K fill | 60 t/s | single stream — curve is flat, not cliffed |
| Speedup vs no-MTP | ~2.6–2.7× at every depth | same config, MTP off |
| NIAH @ 146K | 3/3 | needle retrieved at depth |
| MTP acceptance (predictable text) | ~98% | upper end of observed range |
| MTP acceptance (creative prose) | ~44% → ~48 t/s | lower end — **still 1.9×** |
| TTFT tax from e4m3 KV | ~+10% | vs fp16 KV |

Context ceilings on 2×24 GB (measured): 153,600 (util .93 + prefix cache), 184,000 (util .95, MTP
**off**), 262,144 does not fit with ~37 GB of weights. A parallel NVFP4 build of the same 27B ran
90–92 t/s with ~66% acceptance and fit **3× 151K** concurrent slots at util 0.90 (lighter weights,
bigger KV pool).

## Known limits

- **MTP is the only decode lever here.** Without it, e4m3 + FlashInfer + a TP=2 pair decodes the
  same as a plain baseline (±3%). Don't expect the KV dtype or card choice to move decode.
- **Card-pair bandwidth is decode-neutral at TP=2.** x16+x4 vs x4+x4 made no decode difference —
  TP=2's cross-card traffic is small enough that even an x4 link isn't the bottleneck for decode.
  (This would *not* hold at TP=4 or for prefill-heavy batch.)
- Acceptance — and therefore the actual speedup — depends on how predictable your output is. Report
  it for your workload, not just the headline number.
- `kv_heads = 4` forbids TP=3. Use TP=2 or TP=4.

## Troubleshooting

- **Symptom:** boots fine, dies ~60K tokens into a long prompt (OOM *or* illegal memory access).
  **Cause/fix:** the MTP activation-peak trap. Lower `--gpu-memory-utilization` to ≤ 0.93.
- **Symptom:** `--tensor-parallel-size 3` refuses to start.
  **Cause/fix:** `kv_heads=4` isn't divisible by 3 — use TP=2 or TP=4.
- **Symptom:** MTP seems to do nothing / decode == baseline.
  **Cause/fix:** vLLM didn't find the MTP head. Confirm the head tensors are in the model dir and
  the startup logs acknowledge the speculative config.
- **Symptom:** `--kv-cache-dtype fp8_e5m2` errors on an FP8 checkpoint.
  **Cause/fix:** use `fp8` (e4m3).

## Sources and attribution

- Engine: [vLLM](https://github.com/vllm-project/vllm) — MTP speculative decoding.
- Model: Qwen3.6-27B (INT8 AutoRound and NVFP4 builds), which ship an MTP head.
- Measured on the reference dual RTX 3090 rig, 2026-07. Some large-checkpoint NVFP4 builds that
  quantize `lm_head` needed extra vLLM patches on versions < 0.24.0; prefer ≥ 0.24.0 (see the
  [Ampere FP8/NVFP4 recipe](ampere-fp8-nvfp4-marlin.md)).
