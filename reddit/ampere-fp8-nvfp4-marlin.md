# Title

Yes, you can serve FP8 and NVFP4 checkpoints on a 3090 — vLLM Marlin weight-only kernels, 153/183 t/s. Here's the version gate that decides it

# Body

Full reproducible recipe: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/vllm/ampere-fp8-nvfp4-marlin.md

Common belief: "Ampere can't run FP8/FP4, you need Ada/Hopper." That's true for *native* FP8/FP4
tensor cores — but not for **serving** those checkpoints. vLLM's Marlin kernels keep the weights at
FP8/FP4 in VRAM (the memory/bandwidth win you actually want for decode) and dequantize to fp16
inside the GEMM. A 3090 runs them fine. You forfeit only the native low-precision *compute*
speedup, which mostly matters for prefill/large batch.

**Setup**

- Hardware: 1× RTX 3090 (SM86 / Ampere), CUDA 12.x
- Models: `llmat/Qwen3-4B-Instruct-2507-NVFP4` (compressed-tensors), `Qwen/Qwen3-1.7B-FP8`
- Runtime: vLLM 0.22.1 (**version is load-bearing — see the gate**)

**The settings that mattered**

There is no "use Marlin" flag — vLLM selects it automatically for FP8/FP4 checkpoints on a
no-native-support card. Just `vllm serve <model>` with the quant auto-detected. Two things decide
success:

```text
# 1. ninja must be on PATH or EngineCore dies: FileNotFoundError: 'ninja'
export PATH="$(dirname "$(which vllm)"):$PATH"
# 2. the min-capability GATE for the quant method (SM86 = 86):
#   fp8 / modelopt_fp4(NVFP4) / compressed-tensors W4A4Fp4  -> gate 75  ✅ serves
#   modelopt (FP8) / modelopt_mixed                          -> gate 89  ❌ blocked on Ampere
```

On a correct launch the logs literally say `Using MarlinNvFp4LinearKernel for NVFP4 GEMM` or
`Selected MarlinFP8ScaledMMLinearKernel`. If you don't see a Marlin line, you hit a gate — not a
slow kernel.

**Measured results** (single 3090, single-stream decode, eager)

- NVFP4 (Qwen3-4B-NVFP4): **153 t/s**
- FP8 (Qwen3-1.7B-FP8): **183 t/s**

Small models chosen to prove the *path*, not to benchmark a flagship — the point is these serve at
all, at native-decode-competitive speed.

**The catch**

The two 89-gates (`modelopt` FP8 and `modelopt_mixed`) block many ModelOpt-quantized large
checkpoints on Ampere — but the kernels work, the gate was just set high. That relaxation landed
upstream in **vLLM v0.24.0**, so the clean fix is: upgrade to ≥ 0.24.0 and those paths serve on
Ampere with no patching. Also: weight-only means prefill/batch see less benefit than decode; and use
`--kv-cache-dtype fp8` (e4m3), not `fp8_e5m2`, against FP8 checkpoints.

Recipe with the full gate table, launch commands, and troubleshooting: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/vllm/ampere-fp8-nvfp4-marlin.md

If you've served a large ModelOpt checkpoint on Ampere post-0.24.0, I'd like to hear the model and
your decode numbers.
