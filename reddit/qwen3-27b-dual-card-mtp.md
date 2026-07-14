# Title

Dual-3090 27B with the checkpoint's MTP head for speculative decode: 2.6× decode, flat to 146K context — plus the util trap that OOMs you mid-prefill

# Body

Full reproducible recipe: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/vllm/qwen3-27b-dual-card-mtp.md

Tensor-paralleling a 27B across two 3090s makes it *fit* with long-context headroom, but TP alone
doesn't make decode fast. The lever is the **MTP (multi-token-prediction) head** these Qwen3.6
checkpoints ship: point vLLM's speculative decoder at it and decode jumps ~2.6×, roughly
independent of context depth. The gotcha is a memory-profiling gap that lets a util value boot fine
and then die 60K tokens into a prompt.

**Setup**

- Hardware: 2× RTX 3090 (24 GB) tensor-parallel, Ryzen 9 9900X, no NVLink
- Model: Qwen3.6-27B (INT8 AutoRound and NVFP4 builds; both ship an MTP head)
- Runtime: vLLM 0.22.1, FP8 (e4m3) KV, 151,552 ctx per slot

**The settings that mattered**

```text
--tensor-parallel-size 2 --disable-custom-all-reduce
--kv-cache-dtype fp8
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--mamba-cache-mode align
--gpu-memory-utilization 0.90     # <= 0.93 with MTP on 24GB cards — see the trap
```

MTP proposes 3 tokens/step from the checkpoint's own head; vLLM verifies them in one pass. Without
it, a TP=2 pair decodes like a plain baseline (±3%) — the KV dtype and card choice don't move
decode, MTP does. Note `kv_heads=4`, so TP must be 2 or 4, never 3.

**Measured results** (INT8 build, single stream, depth sweep)

- Decode 0K fill: **79 t/s** → 146K fill: **60 t/s** (flat curve, not a cliff)
- ~**2.6–2.7×** the no-MTP baseline at every depth; NIAH 3/3 at 146K
- MTP acceptance is workload-dependent: ~98% on predictable text, **~44% on creative prose → ~48
  t/s (still 1.9×)**
- e4m3 KV TTFT tax ≈ +10% (not the 2× some reports claim)
- NVFP4 build of the same model: 90–92 t/s, ~66% acceptance, fit **3× 151K** concurrent slots

**The catch (this one cost me two crashes)**

vLLM sizes the KV pool from a startup memory profile that **does not include the MTP drafter's
activation peak**. So util 0.95 launches fine, serves short prompts fine, then dies ~60K tokens into
a long prompt — once as a clean OOM, once as a `CUDA illegal memory access` that *looks* like a
kernel bug but is just memory pressure. Cap util at **≤ 0.93** with MTP on 24 GB cards and it's
stable with prefix caching on.

Recipe with the full launch command, depth-sweep numbers, and troubleshooting: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/vllm/qwen3-27b-dual-card-mtp.md

If you run MTP spec-decode on other checkpoints, I'm curious what acceptance rate you see on your
workload — it swings the real-world speedup a lot.
