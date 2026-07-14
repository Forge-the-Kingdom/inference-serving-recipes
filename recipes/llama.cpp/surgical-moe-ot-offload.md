# Recipe: Surgical MoE expert offload with `-ot` (not `--cpu-moe`)

> **Result (measured):** A 35B-A3B MoE at Q4_K_XL, 96K context, that does **not** fit fully in
> 24 GB, decodes at **75.9 t/s** when only late-layer expert FFNs are offloaded to CPU — versus
> **5.3 t/s** with the binary `--cpu-moe`. Same card, same context, same quant. **14× faster.**

## What this is for

Mixture-of-Experts models are mostly expert-FFN weights. When the whole model won't fit in VRAM
at your target context, the reflex is `--cpu-moe`, which shoves **every** expert FFN to system
RAM. That's a sledgehammer: it frees far more VRAM than you needed and pays DDR bandwidth on
every layer, wasting most of your GPU.

`-ot` (`--override-tensor`) lets you offload **specific tensors** by regex. Offload only the
*late* layers' expert FFNs — just enough to fit — and keep everything else (attention, KV cache,
router, shared experts, early-layer experts, any MTP head) on the GPU. You pay the RAM-bandwidth
penalty on the fewest, lowest-impact tensors instead of all of them.

## Hardware

| Component | Configuration |
|---|---|
| GPU | 1× RTX 3090 (24 GB, Ampere) |
| CPU | AMD Ryzen 9 9900X |
| RAM | 192 GB DDR5-5600 (expert FFNs land here) |
| OS / driver | Linux, NVIDIA 5xx |

Technique is general; the exact split ratio depends on your VRAM and the model.

## Model and runtime

- Model: a MoE GGUF whose experts don't all fit at your target context. Measured on a
  Qwen3.6-35B-A3B (qwen35moe arch, 48 layers, 256 experts × 8 active) at Q4_K_XL.
- Runtime: `llama.cpp` (`llama-server`). Any build with `-ot` / `--override-tensor` works.
- Context: 96K per slot, KV at `q4_0`.
- Concurrency: single slot (`--parallel 1`).

## Launch

```bash
export LLAMA_SERVER="/path/to/llama.cpp/build-cuda/bin/llama-server"
export MODEL_PATH="/path/to/your-moe-model.gguf"

# Push ONLY layers 40–47's expert FFNs to CPU (8 of 48 layers). The "40/8 split".
# Everything else — attention, KV, router, shared experts, layers 0–39 experts — stays on GPU.
OT_REGEX='blk\.(4[0-7])\.ffn_(gate|up|down)_exps\.weight=CPU'

CUDA_VISIBLE_DEVICES=0 "$LLAMA_SERVER" \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 98304 \
  --n-predict -1 \
  --parallel 1 \
  --n-gpu-layers 999 \
  -ot "$OT_REGEX" \
  --batch-size 2048 --ubatch-size 512 \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --reasoning-format auto \
  --metrics
```

### First, confirm your tensor names

The regex above matches Qwen3.6 MoE naming. **Other families differ** — inspect first:

```bash
llama-gguf "$MODEL_PATH" r | grep -i exps | head
```

You're looking for the large per-layer expert FFN tensors (here
`blk.<N>.ffn_(gate|up|down)_exps.weight`). Match *those* by layer, and nothing else.

## Why these settings

- **`-ot 'blk\.(4[0-7])\.ffn_(gate|up|down)_exps\.weight=CPU'`** — the whole recipe. Offloads
  layers 40–47's three expert-FFN matrices to CPU and leaves the rest on GPU. Late layers'
  experts are activated less per token, so concentrating the DDR-read penalty there costs the
  least decode speed. **Tradeoff:** every offloaded layer pays RAM bandwidth on each token; push
  the *fewest* layers that let you boot without OOM.
- **`--n-gpu-layers 999`** — everything not explicitly matched by `-ot` goes to GPU. `-ot` is the
  surgical exception to this blanket rule.
- **`--n-predict -1`** — never cap generation server-side on a reasoning model; a positive cap
  truncates the thinking block.
- **`--cache-type-k/v q4_0`** — quantized KV keeps 96K affordable. Use `q8_0` if you have VRAM
  headroom after the split.
- **`--flash-attn on`, `--ubatch-size 512`** — standard throughput settings; unrelated to the
  offload but part of the measured config.

## Tuning the split

Start aggressive (offload the fewest layers), add more only if it won't fit:

| Split (GPU/CPU) | Regex | Notes |
|---|---|---|
| 40 / 8 | `blk\.(4[0-7])\.` | Most aggressive; try first |
| 32 / 16 | `blk\.(3[2-9]\|4[0-7])\.` | If 40/8 OOMs at boot or prefill |
| 24 / 24 | `blk\.(2[4-9]\|3[0-9]\|4[0-7])\.` | Last resort |

Beyond ~24/24 you're approaching `--cpu-moe` territory — at that point drop context or quant tier
instead of offloading more.

> **llama.cpp gotcha:** recent builds **deprecated multiple `-ot` flags** — a second `-ot`
> silently sends *all* matched experts to the last target. Combine everything into **one**
> comma-separated `-ot` regex.

## Validation

```bash
# Health + a decode-speed read (watch tokens/s in server logs or the /metrics endpoint).
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Write a haiku about VRAM."}],"max_tokens":-1}'
```

Check `nvidia-smi` right after load: VRAM should sit a couple GB under 24 GB (headroom = correct
split). If VRAM is near-empty, you offloaded too much and left the GPU idle — that's the
`--cpu-moe` failure mode in disguise.

## Results

Qwen3.6-35B-A3B Q4_K_XL, 96K ctx, `q4_0` KV, single RTX 3090. Decode throughput, single stream:

| Configuration | Decode t/s | GPU VRAM | Note |
|---|---:|---:|---|
| `--cpu-moe` (all experts to CPU) | 5.3 | 3.7 GB | Wastes ~20 GB of GPU |
| `-ot` 24/24 split | ~50 (est) | ~15 GB | Conservative |
| `-ot` 32/16 split | 30.0 | 18.6 GB | Measured |
| **`-ot` 40/8 split** | **75.9** | 22.4 GB | **Sweet spot** (2 GB headroom) |
| Full GPU, no offload | — | OOM | Won't fit 96K on 24 GB |

Measured 2026-05-26, single-stream decode. The 24/24 row is an estimate; all others are measured.

## Known limits

- **llama.cpp-specific.** vLLM does not expose this tensor-level routing; there, use a smaller
  quant or more cards.
- Offloaded-layer decode is RAM-bandwidth-bound. On slower system memory the penalty per offloaded
  layer is larger, so your sweet-spot split will differ.
- The split is model- and context-specific. Re-tune when you change quant tier or context size.

## Troubleshooting

- **Symptom:** decode barely faster than `--cpu-moe`.
  **Cause/fix:** your regex matched too many layers (or matched attention/shared tensors). Confirm
  it hits only `ffn_(gate|up|down)_exps` on the intended layer range.
- **Symptom:** OOM at boot or during first prefill.
  **Cause/fix:** offload one more band (next row in the tuning table). Prefill peaks above steady
  state, so a config that loads can still OOM on a long first prompt.
- **Symptom:** second `-ot` seems ignored.
  **Cause/fix:** multiple `-ot` flags are deprecated — merge into one comma-separated regex.

## Sources and attribution

- Engine: [llama.cpp](https://github.com/ggml-org/llama.cpp) (`-ot` / `--override-tensor`).
- Model family used for measurement: Qwen3.6-35B-A3B MoE (GGUF quant by Unsloth).
- Technique developed and measured on the reference 4× RTX 3090 rig, 2026-05.
