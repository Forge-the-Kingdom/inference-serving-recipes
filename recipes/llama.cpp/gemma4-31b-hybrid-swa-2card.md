# Recipe: Gemma-4-31B (hybrid SWA) on two 24 GB cards — 256K + MTP + vision, or 2 concurrent seats

> **Result (measured):** A dense 31B multimodal reasoning model at Q4_K_XL serves at **256K
> context, q8_0 KV, with MTP speculative decode AND vision, on two RTX 3090s** — ~40 GB across
> the pair. MTP lifts decode **37.9 → 44.3 t/s** at **79% draft-accept**. Or trade MTP for
> concurrency: **2 seats × 200K context each** at ~42.6 GB. The enabler is the sliding-window
> attention — only 10 of 60 layers grow KV with context.

## What this is for

Gemma-4-31B looks like it shouldn't fit a long context on 48 GB: 17.3 GB of Q4_K_XL weights leaves
~30 GB, and a naive "60 layers × full KV" estimate says 256K KV alone would blow past that. It
doesn't — because the architecture is **hybrid sliding-window attention**: 50 of the 60 layers are
sliding-window (capped at a 1024-token window, so their KV is *constant* regardless of context),
and only **10 full-attention layers** grow KV with sequence length. At 256K/seat, q8_0 KV is
~9.4 GB, not the ~40 GB the naive math predicts.

That headroom is what lets you add **MTP speculative decode** and **vision** and still fit.

## Hardware

| Component | Configuration |
|---|---|
| GPU | 2× RTX 3090 (24 GB each, SM86); one is a 3090 Ti |
| GPU topology | layer-split across the two cards (PCIe x16 + x1 here — see note) |
| CPU / RAM | AMD Ryzen 9 9900X / 192 GB DDR5-5600 |
| OS / driver | Linux, NVIDIA 5xx |

**On the x1 link:** layer-split (pipeline) only passes a ~hidden-size activation vector between
cards per token (~10 KB), so even a PCIe x1 second card doesn't hurt decode here. This is *not*
tensor-parallel — don't confuse the two.

## Model and runtime

- Model: `gemma-4-31B-it-qat` GGUF, `UD-Q4_K_XL` (17.29 GB). Dense (MoE disabled), 60 layers,
  hybrid SWA (50 sliding @ window 1024 + 10 full-attention), 256K train context. It is a
  **reasoning model** (`<think>`, `thinking=1`) and **multimodal** (ships an `mmproj`).
- Runtime: **`llama.cpp` — a recent build.** A brand-new arch like `gemma4` will fail to load on
  an older binary with `unknown architecture`; confirm your build knows it:
  `strings "$LLAMA_SERVER" | grep -c gemma4` (want a nonzero count).
- KV: `q8_0` (the hybrid arch makes this affordable even at 256K).

```bash
# one-time download (uses hf_transfer/hf_xet; verify the sha256 against the repo's LFS oid)
hf download unsloth/gemma-4-31B-it-qat-GGUF \
  gemma-4-31B-it-qat-UD-Q4_K_XL.gguf mmproj-F16.gguf 'MTP/*' --local-dir ./gemma-4-31B-it-qat
```

## Launch — Config A: 1 seat · 256K · MTP · vision

```bash
export LLAMA_SERVER="/path/to/llama.cpp/build/bin/llama-server"   # a build that knows gemma4
export DIR="./gemma-4-31B-it-qat"

CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID "$LLAMA_SERVER" \
  --model "$DIR/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --n-predict -1 \
  --parallel 1 \
  --n-gpu-layers 999 \
  --split-mode layer --tensor-split 1,1 --main-gpu 0 \
  --batch-size 2048 --ubatch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --mmproj "$DIR/mmproj-F16.gguf" \
  --spec-type draft-mtp \
  --model-draft "$DIR/MTP/mtp-gemma-4-31B-it-Q8_0.gguf" \
  --gpu-layers-draft 999 --spec-draft-n-max 2 \
  --jinja --reasoning-format auto \
  --metrics
```

## Launch — Config B: 2 seats · 200K each · vision on CPU · no MTP

```bash
# --ctx-size is the TOTAL pool; --parallel 2 gives each slot 409600/2 = 204800.
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID "$LLAMA_SERVER" \
  --model "$DIR/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 409600 --parallel 2 \
  --n-predict -1 \
  --n-gpu-layers 999 --split-mode layer --tensor-split 1,1 --main-gpu 0 \
  --batch-size 2048 --ubatch-size 512 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --mmproj "$DIR/mmproj-F16.gguf" --no-mmproj-offload \
  --jinja --reasoning-format auto --metrics
```

## Why these settings

- **`--split-mode layer` (not tensor-parallel).** Layer-split fits the weights across both cards
  with minimal cross-card traffic. TP would hammer the slow inter-card link every layer.
- **`--cache-type-k/v q8_0`.** Affordable because only the 10 full-attention layers grow. `q8_0`
  keeps quality; the SWA arch removes the usual reason to drop to `q4_0`.
- **MTP (`--spec-type draft-mtp` + `--model-draft` + `--gpu-layers-draft 999`).** The repo ships a
  trained MTP draft head; `draft-mtp` speculative decode measured **79% acceptance** and **+17%
  decode**. Put the tiny draft on GPU (`--gpu-layers-draft 999`). **MTP requires `--parallel 1`** —
  it cannot run across parallel slots.
- **`--spec-draft-n-max 2`.** Draft-token depth. Newer llama.cpp renamed the old `--draft-max` to
  `--spec-draft-n-max`; the old flag errors.
- **Vision: `--mmproj`, plus `--no-mmproj-offload` in Config B.** Keeping the projector on CPU
  frees ~1.2 GB VRAM; image *encode* gets slower but decode/text is unaffected — the right trade
  when you need the VRAM for a second seat.
- **`--n-predict -1`.** Never cap a reasoning model server-side.

## The core tradeoff: seats vs MTP vs context

**MTP and multiple seats are mutually exclusive** (MTP needs `--parallel 1`). Pick one axis:

| Goal | Config | Per-seat ctx | MTP | ~VRAM |
|---|---|---|---|---|
| Max context + speed, one user | A | 256K | ✅ 79% accept | ~40 GB |
| Concurrency | B | ~200K × 2 | ❌ | ~42.6 GB |
| More concurrency | B | ~128K × 3 | ❌ | ~41 GB (est) |

## Validation

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Capital of Australia, one sentence."}]}'
# at load, check the printed VRAM and, for MTP, the draft-accept ratio in /metrics or timings.
```

`nvidia-smi` after load should show ~20 GB/card (Config A/B). If a long first prompt OOMs, trim
context — prefill peaks above steady state.

## Results

Two RTX 3090s (one Ti), layer-split, q8_0 KV. Measured 2026-07-16.

| Config | Seats × ctx | MTP | Decode t/s | VRAM (GPU0 / GPU1) |
|---|---|---|---:|---|
| A | 1 × 256K | off | 37.9 (shallow) → 21.0 @82K | 15.2 / 15.4 GB |
| **A** | **1 × 256K** | **on (79% accept)** | **44.3** | **~19 / ~21 GB** |
| B | 2 × 200K | off | 38.0 (per stream) | 21.2 / 21.4 GB |

- MTP: **171/216 draft tokens accepted (79%)**, decode 37.9 → 44.3 t/s (+17%).
- Vision verified both configs (correctly read a test image; Config B with the projector on CPU
  kept VRAM flat during encode).
- Config B concurrency: two requests returned in parallel; retrieval correct on an 82K-token
  prompt (KV pre-allocated, VRAM flat through prefill).

## Known limits

- **Needs a current llama.cpp** — `gemma4` is new; older builds error `unknown architecture`.
- MTP is single-seat only. Config B's ~2.5 GB/card headroom held through concurrent text + a
  CPU-vision request, but two *simultaneous* cold 256K fills with images would be the stress edge —
  drop to ~180K/seat for margin, or shift weight with `--tensor-split 0.52,0.48`.
- vLLM/ik add nothing here: the model is a GGUF dense arch (not vLLM-loadable as GGUF), and there
  are no MoE experts to offload.

## Sources and attribution

- Engine: [llama.cpp](https://github.com/ggml-org/llama.cpp) (`--spec-type draft-mtp`, `--mmproj`).
- Model: [unsloth/gemma-4-31B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF).
- Measured on the reference 2× RTX 3090 pair, 2026-07-16.
