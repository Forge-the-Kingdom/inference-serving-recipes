# Recipe: Serving FP8 and NVFP4 checkpoints on Ampere (RTX 3090) via vLLM Marlin

> **Result (measured):** RTX 3090 (SM86) has **no native FP8 or FP4 compute**, yet FP8 and NVFP4
> checkpoints serve on it *today* through vLLM's Marlin weight-only kernels — **153 t/s** (NVFP4)
> and **183 t/s** (FP8) single-stream decode on one 3090, coherent output. You keep the
> memory/bandwidth win; you give up only the native low-precision *compute* speedup.

## What this is for

More and more checkpoints ship pre-quantized to FP8 or NVFP4 (NVIDIA's 4-bit format). The common
belief is "Ampere can't run those — you need Ada/Hopper." That's true for *native* FP8/FP4 tensor
cores, but not for **serving** them. vLLM's Marlin kernels store the weights at FP8/FP4 in VRAM
(the whole point, for memory-bound decode) and dequantize to fp16 *inside* the GEMM. So a 3090 runs
these checkpoints fine — you lose only the compute-side speedup that matters for prefill/batch-heavy
work, not the VRAM footprint or the decode bandwidth advantage.

## Hardware

| Component | Configuration |
|---|---|
| GPU | 1× RTX 3090 (24 GB, SM86 / Ampere) |
| CPU | AMD Ryzen 9 9900X |
| OS / driver | Linux, NVIDIA 5xx, CUDA 12.x |

Any Ampere card (SM80/SM86/SM89-and-below-native-FP8) benefits. The version gates below are what
matter more than the exact card.

## Model and runtime

- Models measured:
  - NVFP4: [`llmat/Qwen3-4B-Instruct-2507-NVFP4`](https://huggingface.co/llmat/Qwen3-4B-Instruct-2507-NVFP4)
    (compressed-tensors)
  - FP8: [`Qwen/Qwen3-1.7B-FP8`](https://huggingface.co/Qwen/Qwen3-1.7B-FP8)
- Runtime: **vLLM 0.22.1**. **Version is load-bearing** — see the capability-gate table below.
- Precision path: weights FP8/FP4 in VRAM → Marlin dequantizes to fp16 in the GEMM → activations
  fp16 (W4A16 / W8A16 effectively). W4A4 checkpoints are accepted and run as W4A16.

## Launch

```bash
export VLLM="/path/to/venv/bin/vllm"
# ninja MUST be on PATH or the engine dies with FileNotFoundError: 'ninja'
export PATH="$(dirname "$VLLM"):$PATH"

# NVFP4 (compressed-tensors) — vLLM auto-detects the quant from the checkpoint config
"$VLLM" serve llmat/Qwen3-4B-Instruct-2507-NVFP4 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# FP8 — same idea; --quantization fp8 is auto-detected from the checkpoint
"$VLLM" serve Qwen/Qwen3-1.7B-FP8 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

On a correct launch the logs explicitly announce the Marlin path:

- NVFP4: `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`
- FP8: `Selected MarlinFP8ScaledMMLinearKernel` + a "no native FP8 → weight-only via Marlin" warning

If you *don't* see a Marlin kernel line, the quant was rejected by a capability gate (below), not
served slowly.

## Why these settings

- **vLLM ≥ 0.22.1** — earlier prod builds (e.g. 0.19.1) have stricter capability gates that reject
  these quants on SM86 outright. The kernels exist; the gate is what changed.
- **Marlin is automatic** — you don't pass a "use Marlin" flag. vLLM sees an FP8/FP4 checkpoint on a
  no-native-support card and selects the Marlin weight-only kernel. Your job is just to not be on a
  version whose gate blocks it.
- **`ninja` on `PATH`** — Marlin/inductor JIT-compiles kernels at startup; without `ninja` the
  EngineCore dies immediately with `FileNotFoundError: 'ninja'`.
- **`--gpu-memory-utilization 0.90`** — leave headroom; see the co-tenancy trap below.
- **Tradeoff:** weight-only means you keep the memory/bandwidth win (great for single-stream decode)
  but forfeit native-FP8/FP4 *compute*. Prefill and large batches are dequant-bound, so this shines
  for decode-heavy / low-concurrency serving, less so for throughput-max batch jobs.

## Capability gates (know before you launch)

Whether a quant serves depends on its **minimum-capability gate** in your vLLM version. Measured in
0.22.1 (`vllm/model_executor/layers/quantization/`), `min_capability` is SM×10 (SM86 = 86):

| Quant method | Gate | Serves on 3090 (SM86)? |
|---|---:|---|
| `fp8` | 75 | ✅ |
| `modelopt_fp4` (NVFP4, incl. W4A16_NVFP4) | 75 | ✅ |
| compressed-tensors W4A4Fp4 / W4A16Fp4 | 75 | ✅ |
| modelopt MXFP8 | 80 | ✅ |
| `modelopt` (FP8, ModelOpt path) | **89** | ❌ blocked |
| `modelopt_mixed` (mixed-precision) | **89** | ❌ blocked |

The last two are what block many ModelOpt-quantized large checkpoints on Ampere — the **kernels
work**, the gate is just set to 89.

> **The good news:** those two 89-gates were fixed upstream. The identical two-line relaxation
> (min-capability 89 → 80, plus a missing `orig_dtype` attribute the Marlin fallback reads) landed
> in **vLLM v0.24.0** (mid-2026), and a follow-up extended the mixed gate down to SM75. **So the
> clean fix is: upgrade to vLLM ≥ 0.24.0** and these ModelOpt paths serve on Ampere with no local
> patching. Only older venvs need a manual gate edit.

## Validation

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","prompt":"The capital of France is","max_tokens":16}'
```

- Startup logs name a `Marlin*` kernel (see above).
- Output is coherent (not garbage — a wrong dequant path produces noise, not errors).
- `nvidia-smi` shows the model resident at roughly its quantized size, not its fp16 size.

## Results

Single RTX 3090, single-stream decode, eager mode, measured 2026-07-07:

| Checkpoint | Quant | Decode t/s | Kernel |
|---|---|---:|---|
| Qwen3-4B-Instruct-2507-NVFP4 | NVFP4 (compressed-tensors) | 153 | `MarlinNvFp4LinearKernel` |
| Qwen3-1.7B-FP8 | FP8 | 183 | `MarlinFP8ScaledMMLinearKernel` |

These are small models chosen to *prove the path*, not to benchmark a flagship — the point is that
FP8/NVFP4 **serve at all** on Ampere, at native-decode-competitive speed. Larger ModelOpt
checkpoints serve the same way once past the 89-gate (upgrade to ≥0.24.0).

## Known limits

- **Weight-only.** No native FP8/FP4 compute — prefill and batch-heavy throughput see less benefit
  than decode.
- **KV-cache dtype:** with FP8-quantized checkpoints, request FP8 KV as `fp8` (e4m3), **not**
  `fp8_e5m2` — the latter errors against these checkpoints.
- **ModelOpt FP8 (pure `modelopt` path)** still carried an 89-gate as of mid-2026 even after the
  mixed-precision fix; if you hit it, upgrade vLLM and re-check.

## Troubleshooting

- **Symptom:** engine dies at startup with `FileNotFoundError: 'ninja'`.
  **Cause/fix:** put the venv's `bin` on `PATH` so `ninja` resolves (Marlin JIT needs it).
- **Symptom:** quant rejected — "not supported on this device / capability".
  **Cause/fix:** you hit an 89-gate (`modelopt` / `modelopt_mixed`). Upgrade to vLLM ≥ 0.24.0.
- **Symptom:** launching vLLM at high util next to an already-running server crashes the *other*
  server.
  **Cause/fix:** co-tenancy hazard — a runtime CUDA alloc under memory pressure can abort a neighbor
  (e.g. a resident llama.cpp). Don't squeeze vLLM into another process's leftover VRAM; free the card
  first, or cap util low enough that both fit.
- **Symptom:** `--kv-cache-dtype fp8_e5m2` errors.
  **Cause/fix:** use `fp8` (e4m3) against FP8-quantized checkpoints.

## Sources and attribution

- Engine: [vLLM](https://github.com/vllm-project/vllm) — Marlin weight-only FP8/FP4 kernels; the
  FP8-via-Marlin weight-only path shipped in an upstream PR merged early 2026; the SM89→SM80
  ModelOpt gate relaxation landed in v0.24.0.
- Models: `llmat/Qwen3-4B-Instruct-2507-NVFP4`, `Qwen/Qwen3-1.7B-FP8`.
- Measured on the reference RTX 3090 rig, 2026-07.
