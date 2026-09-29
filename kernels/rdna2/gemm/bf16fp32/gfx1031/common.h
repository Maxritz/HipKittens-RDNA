/**
 * @file common.h
 * @brief Shared types for the gfx1031 (RDNA2) GEMM optimization ladder.
 *
 * This ladder mirrors gfx1250's structure but targets GFX1031 hardware:
 * - Wave64 threads (not Wave32)
 * - No MFMA instructions; bf16 matmul via v_pk_fmac_f32 packed fma chains
 * - No FP8 support (static_assert'd out)
 * - Unified s_waitcnt (no split counters)
 * - 64KB LDS (not 256KB banked)
 * - No workgroup clustering (no __cluster_dims__)
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include "kittens.cuh"

namespace gfx1031_gemm {

#ifndef GFX1031_ELEM
#define GFX1031_ELEM bf16
#endif
using elem_t = kittens::GFX1031_ELEM;
static_assert(sizeof(elem_t) == 2, "GFX1031_ELEM must be a 16-bit float type");

using gl_e = kittens::gl<elem_t, -1, -1, -1, -1>;
using gl_c = kittens::gl<elem_t, -1, -1, -1, -1, kittens::ducks::gl_layout::col_major>;

template<int R, int C> using st_e =
    kittens::st<elem_t, R, C, kittens::ducks::st_shape::st_16x32_padded<>>;

template<int R, int C> using rt_e =
    kittens::rt<elem_t, R, C, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;

using A_tile = st_e<BLOCK_M, K_STEP>;
using B_tile = st_e<BLOCK_N, K_STEP>;

using C_tile = kittens::st<elem_t, BLOCK_M, BLOCK_N,
                           kittens::ducks::st_shape::st_16x32_padded<128, 8>>;

inline void require_k_blocks(int K, const char* rung)
{
    if (K / BLOCK_K >= 1) return;
    std::fprintf(stderr, "%s: K=%d is shorter than one block of BLOCK_K=%d. Refusing.\n",
                 rung, K, BLOCK_K);
    std::abort();
}

template<int WARP_DIM>
__device__ __host__ constexpr int ds_loads_per_subblock() { return (WARP_DIM / 16) * 2; }

struct gemm_globals {
    gl_e a, b;
    gl_c c;
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
};

struct launch_config {
    hipStream_t stream;
    dim3 grid;
    dim3 block;

    explicit launch_config(const gemm_globals& g, hipStream_t launch_stream = 0)
        : stream(launch_stream),
          grid(g.M() / BLOCK_M, g.N() / BLOCK_N),
          block(NUM_THREADS) {}

    template <int STAGES = 2>
    size_t dynamic_shared_memory() const { return STAGES * (sizeof(A_tile) + sizeof(B_tile)); }
};

} // namespace gfx1031_gemm