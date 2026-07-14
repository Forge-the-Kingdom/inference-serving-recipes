# inference-serving-recipes

Reproducible recipes for serving local LLMs on consumer/prosumer multi-GPU hardware.

Each recipe is a self-contained Markdown file: the exact launch command, an explanation
of *why* each non-obvious flag matters, the tradeoff it buys, validation commands, and
**measured** results with the run conditions attached. Numbers are labeled *measured*,
*observed*, or *estimated* — never blurred together.

Most of this was learned the expensive way on a 4× RTX 3090 box. The techniques are not
specific to that hardware; the constants are. Reproduce on your own rig and send a PR with
your runtime version, context, concurrency, and prompt/decode split so the comparison stays
honest.

## Reference hardware

Unless a recipe says otherwise, results were measured on:

| Component | Configuration |
|---|---|
| GPU | 4× RTX 3090 (24 GB each, SM86 / Ampere); one is a 3090 Ti |
| GPU topology | PCIe x16 / x1 / x4 / x4 — deliberately bandwidth-asymmetric |
| CPU | AMD Ryzen 9 9900X (12C/24T, 2 CCDs × 6 cores) |
| RAM | 192 GB DDR5-5600 |
| OS | Linux (Ubuntu-family), NVIDIA driver 5xx |

The x1 link and the split-CCD CPU are not accidents — a couple of these recipes exist
*because* of that asymmetry, and they'll help anyone else on a mixed-bandwidth board.

## Recipes

### llama.cpp
- [Surgical MoE expert offload with `-ot`](recipes/llama.cpp/surgical-moe-ot-offload.md) —
  push only late-layer expert FFNs to CPU instead of the binary `--cpu-moe`. **14× faster
  decode** (5.3 → 75.9 t/s) at the same VRAM budget.
- [CCD pinning for concurrent MoE offload](recipes/llama.cpp/ccd-pinning-concurrent-moe.md) —
  on a multi-CCD Ryzen, pin each server to one CCD. Recovers **4.3× concurrent decode**
  (5.8 → 25.2 t/s) with `taskset` and `--threads`, zero cost.

### vLLM
- [Serving FP8 / NVFP4 checkpoints on Ampere](recipes/vllm/ampere-fp8-nvfp4-marlin.md) —
  3090s have no native FP8/FP4 compute, but vLLM's Marlin weight-only kernels serve these
  checkpoints today. **153 t/s (NVFP4) / 183 t/s (FP8)** on a single 3090.
- [Dual-card 27B with MTP speculative decode](recipes/vllm/qwen3-27b-dual-card-mtp.md) —
  MTP n=3 + FP8 (e4m3) KV cache on a two-3090 tensor-parallel pair. **~2.6× decode**, flat
  to deep context, plus the memory-profiling trap that OOMs you mid-prefill.

### DwarfStar (`antirez/ds4`)
- [DeepSeek-V4-Flash IQ2, fully GPU-resident on 4× 3090](recipes/dwarfstar/deepseek-v4-iq2-4x3090-gpu-resident.md)
  — an ~80.7 GiB IQ2 checkpoint entirely in VRAM across four cards, **no CPU offload**. Served as a
  **distributed 4-process pipeline** (not a naive in-process layer split), prefill jumps **3.25×**
  on a no-NVLink box: **170 t/s prefill / 23 t/s decode** at 96K context.

## Conventions

- Model and binary paths are environment variables (`$MODEL_PATH`, `$LLAMA_SERVER`, `$VLLM`).
  Set them for your machine.
- Reasoning servers keep the runtime's unlimited-generation convention (`--n-predict -1`,
  `max_tokens: -1`) rather than an invented positive cap — capping starves the thinking block.
- "Measured" means a real run on the reference hardware; conditions (prompt shape, fill depth,
  concurrency, run count) are stated inline. Anything without those is labeled an estimate.

## Contributing

Corrections and cross-hardware reproductions are the whole point. Open an issue or PR with the
result **and the conditions that produced it**. A number without its run conditions can't be
compared and won't be merged.

## License

[MIT](LICENSE) — recipes, commands, and prose. Model weights and engine binaries carry their
own upstream licenses; each recipe links to them.
