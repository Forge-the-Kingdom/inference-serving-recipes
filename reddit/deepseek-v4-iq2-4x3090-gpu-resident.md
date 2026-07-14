# Title

DeepSeek-V4-Flash IQ2 running FULLY GPU-resident on 4× RTX 3090 (no CPU offload): 170 t/s prefill, 23 t/s decode — the win was distributed pipeline, not the quant

# Body

Full reproducible recipe: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/dwarfstar/deepseek-v4-iq2-4x3090-gpu-resident.md

An ~80.7 GiB IQ2XXS DeepSeek-V4-Flash checkpoint fits in 4× 24 GB = 96 GB — barely, with only
~54–118 MiB free per card. But *fitting* it wasn't the hard part. Getting four PCIe 3090s with **no
NVLink** to actually work in parallel was. The naive approach (one process, contiguous layer split,
push each microbatch GPU0→1→2→3 in sequence) gives a sawtooth trace — one card busy at a time,
prefill stuck ~52 t/s.

Serving it as a **distributed pipeline** (antirez's DwarfStar / `ds4`: a coordinator + 3 worker
*processes*, each owning a layer range, multiple 64-token chunks in flight) got all four cards busy
at once (util ≈ 96/58/100/100%) and lifted prefill **3.25×**.

**Setup**

- Hardware: 4× RTX 3090 (96 GB total), Ryzen 9 9900X, **no NVLink**, PCIe only
- Model: DeepSeek-V4-Flash IQ2XXS (routed experts 2-bit; attn/shared/proj/output stay Q8/F16),
  ~86.7 GB, from `antirez/deepseek-v4-gguf`
- Runtime: DwarfStar (`antirez/ds4`), distributed mode, 96K context, fully GPU-resident

**The settings that mattered**

```text
# distributed PP4: coordinator (GPU0, layers 0-9 + embed/output) + 3 workers
--role coordinator/worker  --layers 0:9 / 10:20 / 21:31 / 32:42
--prefill-chunk 64  --dist-prefill-window 5   # small chunks + deep window fill the pipeline
--dist-activation-bits 16
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256            # fit-critical; each worker needs ~16.36 GiB spans
```

Big prefill chunks (512/1024) collapse it back to in-process behavior (~52 t/s). The whole win is
keeping many small chunks in flight across independent processes.

**Measured results** (4096-token benchmark, same weights/cards)

- In-process multi-GPU, chunk 512: **52.4 t/s** prefill / 23.9 decode
- **Distributed PP4, chunk 64: 170.4 t/s** prefill / 23.3 decode ← adopted
- For reference, an llama.cpp UD-IQ1_M path (with 8 expert layers on CPU) did ~97 / 14.6

3.25× prefill over the in-process fork, ~1.6× decode over the CPU-offload llama.cpp baseline, and
it's **entirely in VRAM** — nothing spilled to system RAM.

**The catch**

There's essentially no headroom (~54–118 MiB free per worker at 96K) — raise the arena or context
and it stops fitting. DwarfStar runs only its own matched GGUFs, so engine and checkpoint are a
pair. And this is a **no-NVLink** result specifically; with NVLink the in-process gap would narrow.
This config runs neither DFlash spec-decode nor MTP.

Recipe with the full 4-process launch, fit notes, and troubleshooting: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/dwarfstar/deepseek-v4-iq2-4x3090-gpu-resident.md

If you run this on a 4-GPU box **with** NVLink, I'd love to see whether in-process closes the gap or
distributed still wins.
