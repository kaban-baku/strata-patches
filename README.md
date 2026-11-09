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

### Settings that ran on a 4 GB card (64K context)

Engine arguments and environment that started without paging (the GPU-reach wait in `STRATA_DECODE_TIMING=1` stayed
at ~12 ms; when the card pages it jumps to ~55 ms):

```text
--max-context 65536 --kv q4_0 --spec 2 (no --mtp) --kv-resident 4096 --lookup-max-t 4 --vram-reserve-mib 1112
--resident-budget-gib 24 (or 40 on a 64 GB PC)
STRATA_CPU_HEAD=1  STRATA_HEAD_HOST=1  STRATA_HC_Q8=1  STRATA_PREFILL_LAYER_MAJOR=0
```

- `STRATA_HC_Q8=1` is required: without it the hyper-connection projections load as BF16 (~1.2 GB more) and nothing fits.
- 128K does not fit: with a smaller reserve it starts but pages (9-13 tok/s). Below ~900 MiB of reserve the auto expert
  cache grows into the room the late allocations need and the card pages again.
- The shipped defaults (MTP on) do not start on 4 GB (the expert cache is ~800 MiB short).
- `STRATA_PREFILL_LAYER_MAJOR=0` avoids the layer-major prompt walk on small caches (see
  [#5104](https://github.com/Niko1221/Strata/issues/5104)).

### Results (decode tok/s, 64K, the 4 GB reproduction above)

| model | RAM budget | official v0.2.12 (same settings) | with cpu-head |
|---|---|---:|---:|
| [ISTA GSQ-RCO IQ2_XS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 40 GB | 22.8 | **30.6** |
| [UD-IQ1_M-Fast](https://huggingface.co/kaban-baku/Qwen3.8-Flash-Next-UD-IQ1_M-Fast-GGUF) | 40 GB | 15.5 | **28.4** |
| [UD-IQ1_M-384Exp-Fast](https://huggingface.co/kaban-baku/Qwen3.8-Flash-Next-UD-IQ1_M-384Exp-Fast-GGUF) | 24 GB | 16.9 | **28.4** |

The gain grows with the head's size (Unsloth's Q6_K head is 497 MiB, GSQ's IQ4_XS head 322 MiB). On a clean v0.2.12
tree with only this patch, GSQ IQ2_XS at 24 GB gave 22.2 tok/s (22.5 on the working build that also has
`route-any-ne`). The 384-expert model also needs `route-any-ne` for its full speed. These are reproductions on an
RTX 3070 Ti Laptop with VRAM held by another process, not a real 4 GB card.

## 日本語の要約

VRAM 4GB では出力ヘッドが入らず、エンジンはトークンごとにヘッドを PCIe で読み直しています。cpu-head はそれを CPU で計算するパッチ（`STRATA_CPU_HEAD=1`、オプトイン）です。上の設定（64K・MTP なし・予約 1112MiB など）で、公式 0.2.12 に比べて decode が 1.3〜1.8 倍になりました（8GB のカードを 4GB 相当に制限した再現）。

## License

MIT, the same license as Strata.
