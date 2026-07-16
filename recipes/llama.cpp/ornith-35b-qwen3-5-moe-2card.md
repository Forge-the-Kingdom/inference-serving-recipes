# Recipe: Ornith-1.0-35B (Qwen3.5 hybrid MoE) — Q8 with 2 concurrent seats on two 24 GB cards

> **Result (measured):** A 35B agentic-coding MoE at **Q8_0** (36.9 GB of weights) serves **two
> concurrent seats at 96K context each, q8_0 KV**, on two RTX 3090s using only **~38.4 GB of the
> 48 GB** — ~10 GB to spare. Decode **70–105 t/s** (it's an A3B, ~3B active). Tool-calling works
> out of the box. You do **not** need to drop to Q6.

## What this is for

`Ornith-1.0-35B` is `qwen3_5_moe` (Qwen3.5-35B-A3B). The instinct with a 35B at Q8 is "37 GB of
weights barely fits two cards, forget a long context at high concurrency" — the reflex is to drop
to Q6/Q5. **You don't need to**, because of the attention arch:

- 40 layers, but **30 are `linear_attention`** (Mamba/gated-delta style) with **no growing KV** —
  they carry a small fixed recurrent state, not a per-token cache.
- Only **10 full-attention layers** grow KV, and they use just 2 KV heads.

So 2 seats × 96K of q8 KV is **~2.1 GB total**. The binding constraint is the 36.9 GB of Q8
weights, not KV — which means context and concurrency are nearly free once the weights fit.

## Hardware

| Component | Configuration |
|---|---|
| GPU | 2× RTX 3090 (24 GB each, SM86); one is a 3090 Ti |
| Split | layer-split across both cards |
| CPU / RAM | AMD Ryzen 9 9900X / 192 GB DDR5-5600 |
| OS / driver | Linux, NVIDIA 5xx |

## Model and runtime

- Model: `deepreinforce-ai/Ornith-1.0-35B-GGUF`, `Q8_0` (36.9 GB). Arch `qwen3_5_moe`: 40 layers
  (30 linear-attention + 10 full-attention), 256 experts × 8 active, reasoning model (qwen3
  `<think>`). The GGUF release is **text-only** (no mmproj), despite the model's vision config.
- Runtime: **`llama.cpp` — a current build.** `qwen3_5_moe` (linear-attention + MoE) is very new;
  older binaries won't have it. Confirm: `strings "$LLAMA_SERVER" | grep -c qwen3_5` (want nonzero).
- Q8 weights fit fully in VRAM → **no `-ot` expert offload needed** (unlike the offload recipe).

```bash
hf download deepreinforce-ai/Ornith-1.0-35B-GGUF ornith-1.0-35b-Q8_0.gguf --local-dir ./ornith-1.0-35b
```

## Launch — 2 seats × 96K

```bash
export LLAMA_SERVER="/path/to/llama.cpp/build/bin/llama-server"   # a build that knows qwen3_5_moe

# --ctx-size is the TOTAL pool; --parallel 2 gives each slot 196608/2 = 98304 (96K).
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID "$LLAMA_SERVER" \
  --model ./ornith-1.0-35b/ornith-1.0-35b-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 196608 --parallel 2 \
  --n-predict -1 \
  --n-gpu-layers 999 --split-mode layer --tensor-split 1,1 --main-gpu 0 \
  --batch-size 2048 --ubatch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --reasoning-format auto \
  --metrics
```

## Why these settings

- **Q8_0, not Q6.** Because KV is nearly free on this arch, the full 36.9 GB Q8 weights + 2 seats
  of 96K + buffers land at ~38.4 GB — well under 48 GB. Q6_K (28.5 GB) is the fallback only if you
  also want much larger context/concurrency; at 2×96K it's unnecessary.
- **`--parallel 2 --ctx-size 196608`.** In llama.cpp `--ctx-size` is the shared pool; `--parallel`
  divides it. Size the pool as `seats × per-seat` and keep per-seat ≥ your real prompt depth.
- **`--cache-type-k/v q8_0`.** Affordable — only 10 full-attention layers grow. No reason for `q4_0`.
- **`--split-mode layer`.** Fits weights across both cards with minimal cross-card chatter.
- **`--jinja --reasoning-format auto`.** Drives the GGUF's embedded Qwen3 chat template, which
  includes both the `<think>` reasoning format and the tool-call grammar (see below).
- **No `-ot`.** Everything fits on GPU; expert offload would only slow it down.

## Tool-calling

Ornith is an agentic coding model. With `--jinja`, the standard OpenAI `tools` array works and the
server returns proper `tool_calls` (upstream cards use reasoning-parser `qwen3`, tool-parser
`qwen3_xml` / `qwen3_coder`):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"How many .rs files are in ./src? Use a tool."}],
  "tools":[{"type":"function","function":{"name":"run_shell","description":"Run a shell command",
    "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
  "tool_choice":"auto"}'
```

Measured: `finish_reason: tool_calls`, emitting
`run_shell({"command":"find ./src -name \"*.rs\" | wc -l"})` — correct function, valid JSON args.

## Validation

```bash
# fire two requests at once to exercise both slots
for q in "Write is_prime(n) in Python." "Reverse the string hello."; do
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}]}" & done; wait
```

`nvidia-smi` should show ~19–20 GB/card and stay flat under load (KV is pre-allocated; MoE compute
buffers are tiny here).

## Results

Two RTX 3090s (one Ti), layer-split, Q8_0 weights, q8_0 KV, 2 seats. Measured 2026-07-16.

| Metric | Value |
|---|---|
| VRAM (GPU0 / GPU1) | 20.2 / 19.0 GB (**~38.4 GB / 48**, ~10 GB free) |
| Per-seat context | 96K (`n_ctx=98304` × 2 slots) |
| KV cache (q8_0, 2×96K) | ~2.1 GB (only 10 full-attn layers grow) |
| Decode, single stream | 70–105 t/s (A3B, ~3B active) |
| Concurrency | 2 requests in parallel, VRAM flat |
| Tool-calling | `tool_calls` emitted, correct args, loop closes |

Cross-workload check: a canon-grounded prose generation and an agentic tool-call ran **on the two
seats simultaneously** with VRAM flat.

## Headroom / scaling

KV is ~1 GB per 96K-seat, so the ~10 GB of free VRAM buys a lot: **2 seats × 256K** (adds ~3.5 GB
KV) or **4–6 seats × 96K** also fit. The wall is the 36.9 GB of Q8 weights — scale context and
concurrency freely until you hit it.

## Known limits

- **Needs a current llama.cpp** with `qwen3_5_moe` (linear-attention + MoE) support — an older
  build errors on the arch.
- GGUF release is text-only; the model's vision path isn't available here (no mmproj shipped).
- Numbers are for two 3090s; on cards with more VRAM the same arch scales context/seats further at
  the same near-zero KV cost.

## Sources and attribution

- Engine: [llama.cpp](https://github.com/ggml-org/llama.cpp).
- Model: [deepreinforce-ai/Ornith-1.0-35B-GGUF](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF)
  (Qwen3.5-35B-A3B base).
- Measured on the reference 2× RTX 3090 pair, 2026-07-16.
