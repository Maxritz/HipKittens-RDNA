# gfx1031 GEMM optimization ladder (RDNA2)

`bf16` -> `fp32` GEMM rungs, ordered upward from a naive kernel to the fastest. Each rung adds one
hardware or algorithmic feature and changes nothing else. Accumulation is fp32 throughout.

This ladder mirrors the gfx1250 (CDNA5) ladder but targets **GFX1031 (RDNA2 / RX 6000)** with the following architectural differences:

- **Wave64**: `kittens::WARP_THREADS = 64` (CDNA5 uses 32)
- **No MFMA**: `kittens::mma_ABt` maps to `v_pk_fmac_f32` packed fma chains instead of `v_fma_f32` / matrix instructions
- **No FP8**: static_assert'd out in `include/rdna2/`
- **Unified `s_waitcnt`**: no split counters (`s_wait_loadcnt`, `s_wait_dscnt`, etc.) — uses `s_waitcnt` with combined bitmask
- **64KB LDS**: half of CDNA5's 128KB per CU
- **No workgroup clustering**: `__cluster_dims__` is not supported; cluster rungs (09-12) use workgroup ID swizzling instead
- **No async LDS loads**: LDS fills use direct `ds_store` / `ds_load` paths

## Tile geometry
Each rung declares its own `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `K_STEP`, `WARPS_M` and `WARPS_N`. The macro tile is `BLOCK_M` x `BLOCK_N` of output, split across a `WARPS_M` x `WARPS_N` grid of warps.

`K_STEP` is the matrix instruction's K depth, fixed at 32. `BLOCK_K` is how much K one LDS stage holds.

## Synchronization
- `00_gemm_naive` and `01_gemm_double_buf` use the full barrier `sync::sync` and `wait_load`
- `02_gemm_async` adds `wait_async` for copy engine drains
- `09_gemm_wgc` uses workgroup ID swizzling instead of cluster barriers (no `__cluster_dims__`)

## Benchmark

```
make all-kernels                              # 13 pybind11 modules
python3 test.py                               # every rung, four shapes, against torch.matmul
./gemm_ladder.py -r 25 -i 100 --json-out results.json
```

## Build

Targets `gfx1031`. ROCm 7.2+ supports all rungs. Correctness needs PyTorch with gfx1031 support.

```
make KERNEL=00_gemm_naive
```