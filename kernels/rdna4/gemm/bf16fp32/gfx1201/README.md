# gfx1201 GEMM optimization ladder (RDNA4)

`bf16` -> `fp32` GEMM rungs, ordered upward from a naive kernel to the fastest. Each rung adds one
hardware or algorithmic feature and changes nothing else. Accumulation is fp32 throughout.

This ladder mirrors the gfx1250 (CDNA5) ladder but targets **GFX1201 (RDNA4 / RX 9000)** with the following architectural differences:

- **Wave32**: `kittens::WARP_THREADS = 32` (same as CDNA5)
- **Real WMMA**: `kittens::mma_ABt` maps to `__builtin_amdgcn_wmma_f32_16x16x32_bf16/f16/fp8_fp8`
- **IEEE/OCP FP8**: full FP8 support via WMMA intrinsics (not emulated)
- **Split counters**: `s_wait_loadcnt`, `s_wait_dscnt`, `s_wait_kmcnt`, `s_wait_vmcnt` for finer-grained synchronization
- **256KB banked LDS**: banked access patterns, larger capacity than CDNA5
- **Workgroup clustering**: `__cluster_dims__` is supported, same as CDNA5
- **No scaled FMMA**: no `v_fma_scale_f32` instruction

## Tile geometry
Same as CDNA5: each rung declares its own `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `K_STEP`, `WARPS_M` and `WARPS_N`.

## Synchronization
Uses the split counter model via `kittens::sync::wait_load`, `kittens::sync::wait_ds`,
`kittens::sync::sync`, and cluster barriers via `s_barrier` for the cluster rungs (09-12).

## Benchmark

```
make all-kernels                              # 13 pybind11 modules
python3 test.py                               # every rung, four shapes, against torch.matmul
./gemm_ladder.py -r 25 -i 100 --json-out results.json
```

## Build

Targets `gfx1201`. ROCm 7.2+ supports all rungs including clustering. Correctness needs PyTorch
with gfx1201 support.

```
make KERNEL=00_gemm_naive
```