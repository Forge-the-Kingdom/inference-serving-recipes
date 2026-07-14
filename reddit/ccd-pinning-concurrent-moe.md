# Title

Running two MoE offload servers on one Ryzen? Pin each to its own CCD — 4.3× concurrent decode (5.8 → 25.2 t/s), two flags

# Body

Full reproducible recipe: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/llama.cpp/ccd-pinning-concurrent-moe.md

If you run two `llama.cpp` servers that both offload MoE experts to CPU, and you let them spray
threads across all cores, they thrash each other's L3 cache and context-switch each other to
death. On a chiplet Ryzen (Zen 4/5, multiple CCDs) the fix is to give **each server its own CCD**
and cap threads to that CCD's physical cores. It recovered 4.3× on my box for the cost of a
`taskset` mask and `--threads`.

**Setup**

- Hardware: Ryzen 9 9900X (12C/24T, **2 CCDs × 6 cores**), 2× RTX 3090, 192 GB DDR5
- Two servers, one per GPU, both offloading expert FFNs to CPU

**The settings that mattered**

```text
# Server A → CCD0
taskset -c 0-5,12-17 llama-server ... --threads 6
# Server B → CCD1
taskset -c 6-11,18-23 llama-server ... --threads 6
```

Expert matmul for CPU-offloaded layers is **memory-bandwidth-bound**, not compute-bound — 6
physical cores already saturate a CCD's bandwidth. Adding the SMT siblings (12 threads) *lowers*
throughput; it just adds contention. Derive your own mask from `lscpu -e=CPU,CORE,CACHE`.

**Measured results** (both servers under concurrent load)

- Unpinned (16 + 12 threads on 12 cores): **5.8 t/s** (server A) / 9.2 t/s (server B)
- Server A pinned to CCD0 (`taskset -c 0-5,12-17`, `--threads 6`): **25.2–25.9 t/s** — only −6% vs
  its solo rate
- Pin server B to CCD1 too and its gap closes the same way

Bonus: solo decode was slightly *faster* at 6 threads than 16 (26.9 vs 25.5 t/s). On a
bandwidth-bound workload, more threads ≠ more speed.

**The catch**

Only helps when the bottleneck is CPU-offloaded work — fully-in-VRAM servers don't contend on CPU.
The core→CCD map is chip/BIOS-specific; verify with `lscpu -e` before copying my masks or you'll
pin *across* CCDs and make it worse.

Recipe with the topology-mapping steps and validation: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/llama.cpp/ccd-pinning-concurrent-moe.md

Curious whether Intel P/E-core boxes see the same effect — if you test it, post your `lscpu`
layout and numbers.
