# Recipe: DeepSeek-V4-Flash IQ2, fully GPU-resident on 4× RTX 3090 (DwarfStar distributed pipeline)

> **Result (measured):** An ~80.7 GiB IQ2XXS DeepSeek-V4-Flash checkpoint runs **entirely in
> VRAM across four RTX 3090s — no CPU expert offload** — at **170.4 t/s prefill / 23.3 t/s decode**
> (4096-token benchmark, 96K context). The trick isn't the quant; it's serving it as a
> **distributed 4-process pipeline** instead of a naive in-process layer split, which on a
> no-NVLink box lifts prefill **3.25×** (52 → 170 t/s) by getting real cross-card overlap.

## What this is for

A model whose weights sum to ~80.7 GiB *fits* in 4× 24 GB = 96 GB — barely. The hard part isn't
fitting it, it's making four PCIe-connected 3090s with **no NVLink** actually work in parallel
instead of taking turns. The obvious approach (one process, contiguous layer placement, push each
microbatch through GPU0→1→2→3 in sequence) produces a sawtooth utilization trace: one card busy at
a time, prefill stuck around 52 t/s, because every cross-device hop synchronizes through
pinned-host bounce buffers.

[DwarfStar](https://github.com/antirez/ds4)'s **distributed** mode instead runs one coordinator +
N worker *processes*, each owning a layer range on its own GPU, with multiple small prefill chunks
in flight at once. That keeps all four cards busy simultaneously (measured util ≈ 96/58/100/100%)
and turns the sequential sawtooth into a real pipeline. Fully GPU-resident, no expert FFN spilled
to CPU.

## Hardware

| Component | Configuration |
|---|---|
| GPU | 4× RTX 3090 (24 GB each = 96 GB total), Ampere |
| Interconnect | PCIe, **no NVLink** — this is the whole reason distributed beats in-process |
| CPU | AMD Ryzen 9 9900X |
| RAM | 192 GB DDR5-5600 (not used for weights here — all weights are in VRAM) |
| OS / driver | Linux, NVIDIA 5xx, CUDA 12.x |

## Model and runtime

- Model: a **DeepSeek-V4-Flash IQ2XXS** GGUF from
  [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) — mixed-precision:
  only routed experts are heavily compressed (2-bit, `w2Q2K`); attention, shared experts,
  projections, router, and output stay Q8/F16-class (`AProjQ8-SExpQ8-OutQ8`). On-disk ~86.7 GB
  (80.7 GiB).
- Runtime: [DwarfStar (`antirez/ds4`)](https://github.com/antirez/ds4) — a native DeepSeek-V4
  inference engine with an OpenAI-compatible HTTP API. Build the `ds4-server` (coordinator) and
  `ds4` (worker) binaries per its README. **DwarfStar only runs the GGUFs it ships** — pair the
  engine with the matching checkpoint.
- Context: 98,304 (96K), single slot.
- Placement: pipeline-parallel across all 4 GPUs, fully resident.

## Layout

| GPU | Role | Layers |
|---|---|---|
| GPU0 | coordinator (+ embedding & output head) | 0–9 |
| GPU1 | worker | 10–20 |
| GPU2 | worker | 21–31 |
| GPU3 | worker | 32–42 |

## Launch

Four processes — three workers that retry until the coordinator appears, then the coordinator.
Each worker shares the host, so each needs a **unique lock file**.

```bash
export DS4_SERVER="/path/to/ds4/ds4-server"      # coordinator binary
export DS4_WORKER="/path/to/ds4/ds4"             # worker binary
export MODEL_PATH="/path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256        # fit-critical (see below)

DIST_HOST=127.0.0.1 ; DIST_PORT=19000
CTX=98304 ; PREFILL_CHUNK=64 ; DIST_WINDOW=5

# --- workers: one per non-coordinator GPU ---
start_worker() { # gpu  id  layers
  DS4_LOCK_FILE="/tmp/ds4-worker$2.lock" CUDA_VISIBLE_DEVICES="$1" \
    "$DS4_WORKER" -m "$MODEL_PATH" --cuda --prefill-chunk "$PREFILL_CHUNK" \
      --ctx "$CTX" --role worker --layers "$3" \
      --coordinator "$DIST_HOST" "$DIST_PORT" &
}
start_worker 1 1 10:20
start_worker 2 2 21:31
start_worker 3 3 32:42
sleep 2   # let workers fail visibly before the coordinator commits GPU0's big alloc

# --- coordinator: GPU0, owns the HTTP API ---
DS4_LOCK_FILE="/tmp/ds4-coordinator.lock" CUDA_VISIBLE_DEVICES=0 \
  "$DS4_SERVER" -m "$MODEL_PATH" --cuda --prefill-chunk "$PREFILL_CHUNK" \
    --ctx "$CTX" --tokens 393216 \
    --host 127.0.0.1 --port 8000 \
    --role coordinator --layers 0:9 \
    --listen "$DIST_HOST" "$DIST_PORT" \
    --dist-prefill-chunk "$PREFILL_CHUNK" \
    --dist-prefill-window "$DIST_WINDOW" \
    --dist-activation-bits 16
```

## Why these settings

- **Distributed PP4 (coordinator + 3 workers), not in-process** — the headline. Independent
  processes keep multiple 64-token chunks in flight, so all four cards compute at once. DwarfStar's
  *in-process* multi-GPU fork sends each whole microbatch through every device in turn and
  synchronizes on pinned-host bounce buffers → sequential sawtooth, ~52 t/s prefill. Same weights,
  same cards; the pipeline is 3.25× at prefill.
- **`--prefill-chunk 64` + `--dist-prefill-window 5`** — small chunks with a 5-deep window are what
  fill the pipeline. Big chunks (512/1024) run in-process-style and give up most of the overlap
  (measured 51–52 t/s prefill).
- **`--dist-activation-bits 16`** — transport inter-stage activations at 16-bit; cheap on the wire,
  keeps quality.
- **`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256`** — **fit-critical.** Each worker needs ~16.36 GiB of
  required weight spans; leave model mappings unlimited and set the arena chunk to 256 MiB. Artificial
  14/15 GiB weight-cache caps fail to load. Do not raise the arena or graph size without re-running a
  full 96K allocation test — workers finish with only ~54–118 MiB of VRAM free.
- **`--ctx 98304`** — full 96K. The tight fit is *at* this context; a smaller model or context gives
  you room for a larger arena.
- **Unique `DS4_LOCK_FILE` per process** — required when all workers share one host.
- **Sampling** follows the model card: temperature 1.0, top-p 1.0 (DwarfStar adds min-p 0.05).

## Validation

```bash
# OpenAI-compatible API on the coordinator
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say DS4_READY."}],"max_tokens":-1}'
```

While a long prompt is prefilling, watch `nvidia-smi dmon` (or `nvidia-smi -l 1`): you want **all
four cards busy together** (~96/58/100/100%), not one-at-a-time. A sequential sawtooth means you're
on the in-process path or your chunk size is too large.

## Results

DeepSeek-V4-Flash IQ2XXS, 4× RTX 3090, 96K ctx, fully GPU-resident. 4096-token benchmark, measured
2026-07-13:

| Engine configuration | Prefill t/s | Decode t/s |
|---|---:|---:|
| llama.cpp IQ1_M, 8 expert layers on CPU (`-ot`) — older baseline | ~97 | 14.6 |
| DwarfStar in-process multi-GPU, chunk 512 | 52.4 | 23.9 |
| DwarfStar in-process multi-GPU, chunk 1024 | 51.4 | 24.0 |
| **DwarfStar distributed PP4, chunk 64 (adopted)** | **170.4** | **23.3** |

The distributed path is **3.25×** the in-process fork at prefill and **1.76×** the older llama.cpp
prompt receipt, while keeping essentially all of DwarfStar's ~1.6× decode gain over that baseline.
GPU utilization plateau ≈ 96/58/100/100% (real overlap). All numbers measured; the llama.cpp row is
a different engine/quant, shown for context.

## Known limits

- **Fully GPU-resident is the point *and* the constraint.** ~54–118 MiB free per worker at 96K —
  there is no headroom. Larger arena, graph, or context needs a fresh allocation test and may not fit.
- **Engine ↔ checkpoint are a matched pair.** DwarfStar runs its own GGUFs; you can't point it at an
  arbitrary DeepSeek GGUF, and you can't serve these particular files with stock llama.cpp the same way.
- **No NVLink assumed.** With NVLink the in-process gap would narrow; this recipe's win is specifically
  a PCIe/no-NVLink result.
- **This adopted config runs neither speculative decode (DFlash) nor MTP** — those are a separate
  kernel-development track, not part of this serving path. Don't assume they're on.

## Troubleshooting

- **Symptom:** prefill stuck ~52 t/s, `nvidia-smi` shows one card at a time.
  **Cause/fix:** you're effectively in-process — reduce `--prefill-chunk` to 64 and set
  `--dist-prefill-window 5`; confirm you launched separate worker processes, not one process.
- **Symptom:** a worker fails to load / OOMs at startup.
  **Cause/fix:** an artificial weight-cache cap. Leave mappings unlimited and set
  `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256`; each worker needs ~16.36 GiB of required spans.
- **Symptom:** second worker won't start / lock error.
  **Cause/fix:** give each process a unique `DS4_LOCK_FILE`.
- **Symptom:** `max_tokens: -1` returns zero tokens.
  **Cause/fix:** use an engine build where `-1` is honored at both JSON-parse and decode boundaries
  (fixed upstream); never cap generation on a reasoning model.

## Sources and attribution

- Engine: **DwarfStar** — [`antirez/ds4`](https://github.com/antirez/ds4) (distributed inference,
  CUDA pipeline). Thanks to the DwarfStar project; it credits llama.cpp in turn.
- Model: [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf),
  DeepSeek-V4-Flash IQ2XXS mixed-precision quant.
- Measured on the reference 4× RTX 3090 / Ryzen 9 9900X rig, 2026-07-13.
