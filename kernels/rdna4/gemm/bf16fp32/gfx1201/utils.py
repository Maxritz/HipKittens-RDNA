"""Shared constants and tensor helpers for the gfx1201 (RDNA4) GEMM ladder.

The naming follows gfx1250's ladder but targets RDNA4 hardware:
- Wave32 threads
- Real WMMA intrinsics (__builtin_amdgcn_wmma_f32_16x16x32_*)
- IEEE/OCP FP8 support
- Split counters (s_wait_loadcnt, s_wait_dscnt, s_wait_kmcnt, s_wait_vmcnt)
- 256KB banked LDS
"""

import math

import torch

DTYPE = torch.bfloat16
DEVICE = "cuda:0"

RUNGS = ["00_gemm_naive", "01_gemm_double_buf", "02_gemm_async", "03_gemm_128x128",
         "04_gemm_256x256", "05_gemm_deepk", "06_gemm_segment", "07_gemm_tdm",
         "08_gemm_split_bar", "09_gemm_wgc_multicast", "10_gemm_epilogue", "11_gemm_one_wave",
         "12_gemm_two_waves"]

SHAPES = [
    (8192, 8192, 8192),
    (4096, 8192, 2048),
    (8192, 4096, 2048),
    (2048, 1024, 1024),
]


def init_operand(shape, dtype=DTYPE, device=DEVICE, lo=-3, hi=3):
    return torch.randint(lo, hi + 1, shape, dtype=torch.int8, device=device).to(dtype)


def init_c(m, n, dtype=DTYPE, device=DEVICE):
    return torch.empty_strided((m, n), (1, m), dtype=dtype, device=device).fill_(float("nan"))


def print_title(title, width=30):
    print("-" * width)
    print(title)
    print("-" * width)


def gemm_reference(a, b):
    return torch.matmul(a.float(), b.float().t())


def compare(got, ref, k):
    atol, rtol = 0.5 * math.sqrt(k / 8192.0), 1.0e-2
    got_f = got.float()
    err = (got_f - ref).abs()
    return {
        "bad": int((err > atol + rtol * ref.abs()).sum()),
        "n": ref.numel(),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "max_ref_abs": float(ref.abs().max()),
        "nonfinite": int((~torch.isfinite(got_f)).sum()),
        "atol": atol,
    }