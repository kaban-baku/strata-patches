# strata-patches

Local patches for [Strata](https://github.com/Niko1221/Strata), shared so the numbers in the linked issues can be
reproduced. Each folder is one change, written against a tagged release. None of them changes the default path:
each is opt-in or only affects models the release does not handle yet.

| Folder | What it does | Against | Issue |
|---|---|---|---|
| [`cpu-head/`](cpu-head/) | `STRATA_CPU_HEAD=1`: when the output head lives in pinned host memory (a card too small to keep it in VRAM), the CPU computes the logits from RAM instead of the GPU reading the head over PCIe for every row | v0.2.12 | (to be linked) |
| [`route-any-ne/`](route-any-ne/) | `STRATA_ROUTE_PRIOR` / `STRATA_ROUTE_RESIDENT` and the windowed verify router (`moe_route_window`) on models with fewer than 512 experts | v0.2.12 (and the 0.2.11 base) | [#5287](https://github.com/Niko1221/Strata/issues/5287) |

Apply from the root of a Strata source tree at the matching tag, then build as usual:

```sh
git apply /path/to/strata-patches/cpu-head/cpu_head-v0.2.12.patch
```

Tested on Windows 11, RTX 3070 Ti Laptop 8 GB (CUDA 12.4, sm_86), Ryzen 7 6800H, 64 GB RAM. HIP and SYCL builds are
not tested.

## cpu-head

On a 4 GB card (reproduced on the 8 GB card by holding VRAM in another process so that 3,584 MiB are free) the
engine keeps Qwen3.8-Flash-Next's output head (Q6_K, 497 MiB for the Unsloth files) in pinned host memory and the
GPU reads it over PCIe for every logits row: about 42 ms per row here, most of a decode step. The CPU reads the same
bytes from RAM at ~47 GB/s (8 threads), so this patch rings the host once more after the last layer of a verify
window: the GPU publishes the head's input rows to mapped memory, the host computes the logits with ggml-cpu's dot
products into mapped memory, and the GPU copies them back and continues (argmax, sampling, logit bias unchanged).

- Opt-in: `STRATA_CPU_HEAD=1`, used only when the head is already in host memory (the engine moved it there, or
  `STRATA_HEAD_HOST=1`) and the verifier runs the whole model on one GPU. `STRATA_CPU_HEAD_THREADS` sets the thread
  count (default 8). Batch windows keep the GPU path.
- New files: `src/core/cpu_head.cpp`, `include/strata/core/cpu_head.hpp`; changes in `src/core/verify.cpp`.

Measurements and the comparison with an unmodified v0.2.12 are in the issue.

## License

MIT, the same license as Strata.
