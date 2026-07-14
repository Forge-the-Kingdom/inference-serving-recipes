# Recipe: CCD pinning for concurrent MoE offload on multi-CCD Ryzen

> **Result (measured):** Two MoE servers doing CPU expert-offload on one Ryzen 9 9900X
> (2 CCDs × 6 cores). Unpinned, they thrash each other down to **5.8 t/s**. Pin each server to
> its own CCD with `taskset` + `--threads 6` and the same server runs at **25.2 t/s** — **4.3×**,
> for the cost of two flags.

## What this is for

If you run two (or more) `llama.cpp` servers that each offload MoE expert FFNs to CPU, they fight
over the same physical cores and the same L3 cache. On a chiplet Ryzen (Zen 4/5), cores are grouped
into CCDs, each with its own L3 slice. Let both servers spray threads across all cores and you get
cross-CCD L3 thrashing plus context-switch overhead that dwarfs the actual matmul work.

The fix is to **give each server its own CCD** and cap its thread count to that CCD's physical
cores. CPU-offloaded expert matmul is memory-bandwidth-bound, not compute-bound — 6 physical cores
already saturate a CCD's bandwidth, so more threads only add contention.

## Hardware

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen 9 9900X — 12 cores / 24 threads, **2 CCDs × 6 cores** |
| GPU | 2× RTX 3090 (one server per card, both offloading experts to CPU) |
| RAM | 192 GB DDR5-5600 |

Applies to any multi-CCD Ryzen (3900X/5900X/7900X/9900X, Threadripper) running ≥2 offload servers.

### Find your CCD → logical-core map

```bash
lscpu -e=CPU,CORE,SOCKET,NODE | head -30
# or, for the L3-sharing groups directly:
lscpu -e=CPU,CORE,CACHE
```

On the 9900X the layout is:

- **CCD0** — physical cores 0–5 → logical CPUs `0-5,12-17` (12–17 are the SMT siblings)
- **CCD1** — physical cores 6–11 → logical CPUs `6-11,18-23`

Confirm on *your* chip; the mapping varies by model and BIOS.

## Launch

Two servers, one pinned per CCD. Each gets 6 threads = its CCD's physical cores.

```bash
export LLAMA_SERVER="/path/to/llama.cpp/build-cuda/bin/llama-server"

# --- Server A: GPU 0, pinned to CCD0 (cores 0-5,12-17) ---
CUDA_VISIBLE_DEVICES=0 taskset -c 0-5,12-17 "$LLAMA_SERVER" \
  --model "$MODEL_A" \
  --host 127.0.0.1 --port 8081 \
  --ctx-size 98304 --n-predict -1 --parallel 1 \
  --n-gpu-layers 999 -ot "$OT_REGEX_A" \
  --threads 6 \
  --flash-attn on --metrics &

# --- Server B: GPU 1, pinned to CCD1 (cores 6-11,18-23) ---
CUDA_VISIBLE_DEVICES=1 taskset -c 6-11,18-23 "$LLAMA_SERVER" \
  --model "$MODEL_B" \
  --host 127.0.0.1 --port 8082 \
  --ctx-size 98304 --n-predict -1 --parallel 1 \
  --n-gpu-layers 999 -ot "$OT_REGEX_B" \
  --threads 6 \
  --flash-attn on --metrics &
```

(For the `-ot` expert-offload regex itself, see the
[surgical MoE offload recipe](surgical-moe-ot-offload.md).)

## Why these settings

- **`taskset -c 0-5,12-17`** — hard-pins the process to one CCD's logical cores. Its threads never
  migrate onto the other CCD, so the two servers stop sharing an L3 slice. This is the single change
  that buys the 4.3×.
- **`--threads 6`** — one thread per physical core of the CCD. Because the work is
  bandwidth-bound, adding more threads (the measured comparison was 6 vs 16) *reduces* throughput —
  it just adds contention on the same cores' bandwidth. Include the SMT logical IDs in the `taskset` mask
  anyway so the OS keeps helper threads local.
- **Tradeoff:** you're partitioning the CPU. Each server gets half the box. That's exactly what you
  want for two co-resident servers, but if you sometimes run only one, drop the pin and give it more
  cores (see the solo note below).

## Validation

Run both servers under concurrent load and read decode t/s from each `/metrics` (or server logs).
Then kill one and read the survivor's solo rate. Pinned, concurrent should land within ~10% of solo:

```bash
# hammer both at once, compare tokens/s to each server's solo number
for p in 8081 8082; do
  curl -s http://127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"local","messages":[{"role":"user","content":"Count to 200."}],"max_tokens":-1}' &
done; wait
```

## Results

Two servers, concurrent load, measured on the reference rig (2026-06):

| Config | Server A (122B MoE) | Server B (35B MoE) |
|---|---:|---:|
| **Unpinned** (16 + 12 threads on 12 cores) | 5.8 t/s (−78% vs solo) | 9.2 t/s (−93% vs solo) |
| **A pinned to CCD0** (`taskset -c 0-5,12-17`, `--threads 6`) | **25.2–25.9 t/s** (−6% vs solo) | 89.8 t/s (B still unpinned, −33%) |

Pinning **both** servers (B → `6-11,18-23`) closes B's gap too. Solo baselines: A ≈ 26.9 t/s,
B ≈ 135 t/s.

**Bonus finding:** solo decode is *slightly faster* at 6 threads than at 16 (26.9 vs 25.5–25.8 t/s) —
fewer threads = less intra-process contention on the same L3. On a bandwidth-bound workload, more
threads is not more speed.

## Known limits

- Helps only when the bottleneck is **CPU-offloaded** work. Fully-in-VRAM servers don't contend on
  CPU and won't benefit.
- The core→CCD map is chip- and BIOS-specific. Verify with `lscpu -e` before copying the masks.
- Batch/async workloads (e.g. background "dreamer" servers) can share the leftover SMT siblings or
  just tolerate contention — they don't need a dedicated CCD.

## Troubleshooting

- **Symptom:** pinning didn't help / made it worse.
  **Cause/fix:** wrong core mask — you pinned across CCDs. Re-derive from `lscpu -e=CPU,CORE,CACHE`
  and make sure each server's mask stays inside one L3 group.
- **Symptom:** one server is fast, the other still slow.
  **Cause/fix:** only one is pinned. Pin both, to *different* CCDs.
- **Symptom:** solo throughput dropped after adding `--threads 6`.
  **Cause/fix:** it shouldn't for offload work — but if this server is compute-bound (little/no CPU
  offload), give it more threads; the 6-thread cap is specifically for bandwidth-bound expert offload.

## Sources and attribution

- Engine: [llama.cpp](https://github.com/ggml-org/llama.cpp).
- `taskset` (util-linux) for CPU affinity; CCD topology from `lscpu`.
- Measured on the reference Ryzen 9 9900X + dual RTX 3090 rig, 2026-06.
