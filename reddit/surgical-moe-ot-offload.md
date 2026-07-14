# Title

14× faster MoE decode on a single 3090 by offloading only *late-layer* experts with `-ot` (not `--cpu-moe`): 5.3 → 75.9 t/s

# Body

Full reproducible recipe: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/llama.cpp/surgical-moe-ot-offload.md

If you run a MoE that doesn't quite fit in 24 GB at your target context, the reflex is
`--cpu-moe`. Don't. That flag pushes *every* expert FFN to system RAM, freeing far more VRAM
than you needed and paying DDR bandwidth on every layer. Using `-ot` (`--override-tensor`) to
offload **only the late layers' expert FFNs** — just enough to fit — was 14× faster at the same
VRAM budget on my box.

**Setup**

- Hardware: 1× RTX 3090 (24 GB), Ryzen 9 9900X, 192 GB DDR5-5600
- Model: Qwen3.6-35B-A3B, Q4_K_XL GGUF (256 experts × 8 active, 48 layers)
- Runtime: llama.cpp (`llama-server`), CUDA build
- Context: 96K, `q4_0` KV, single slot

**The settings that mattered**

```text
# Offload ONLY layers 40–47's expert FFNs to CPU. Everything else stays on GPU.
-ot 'blk\.(4[0-7])\.ffn_(gate|up|down)_exps\.weight=CPU'
--n-gpu-layers 999
```

Late layers' experts activate less per token, so concentrating the RAM-bandwidth penalty there
costs the least decode speed. You tune one number — how many late layers to push — for the
fewest that let you boot without OOM.

**Measured results** (single-stream decode, same card / context / quant)

- `--cpu-moe` (all experts to CPU): **5.3 t/s**, 3.7 GB VRAM used (~20 GB wasted)
- `-ot` 32/16 split: **30.0 t/s**, 18.6 GB
- `-ot` 40/8 split: **75.9 t/s**, 22.4 GB ← sweet spot, 2 GB headroom
- Full GPU, no offload: OOM (won't fit 96K on 24 GB)

**The catch**

Tensor names are family-specific — inspect with `llama-gguf <model> r | grep exps` before copying
the regex; you want the big `ffn_(gate|up|down)_exps` matrices and nothing else. Also, recent
llama.cpp **deprecated multiple `-ot` flags** — a second one silently sends all experts to the last
target. Combine into one comma-separated regex. The split is memory-bandwidth-bound, so your sweet
spot shifts with RAM speed.

Recipe, tuning table, and validation steps: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/llama.cpp/surgical-moe-ot-offload.md

If you reproduce on different hardware, please share your RAM speed, the split you landed on, and
the model — the optimum moves with memory bandwidth.
