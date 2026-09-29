/**
 * @file 11_gemm_one_wave.cpp
 * @brief Rung 11 -- 10 with one wave per SIMD (RDNA2 (gfx1031)).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        128x128 macro, 32x32 per warp, 2x2 warps; BLOCK_K 32 = 1 x K_STEP 32
 *   occupancy   2 warps / 64 threads / 1 wave per SIMD (Wave64); up to 8 workgroups per CU
 *   registers   ~120 VGPR; 32 are accumulator
 *   spills      none: 0 VGPR, 0 scratch
 *   LDS         2 stages x ~4.2 KB = ~8.4 KB of 64 KB (13%)
 *   sync        per K-block: 1 barrier, 2 LDS drains
 *   intensity   32 FLOP per byte of global operand traffic
 *
 * RDNA2 (gfx1031). This rung takes rung 10's 256x256 tile, halves it to 128x128 (2x2 warps, 64 threads),
 * trading occupancy for register pressure headroom. The 64-thread workgroup fits one Wave32 on RDNA2
 * or one Wave64 on RDNA2 (mapped to one wave). No TDM or cluster is used.
 */

#include "kittens.cuh"

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 128;
constexpr int BLOCK_K     = 32;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 2;
constexpr int WARPS_N     = 2;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int K_SUBBLOCKS = BLOCK_K / K_STEP;

#include "common.h"

using namespace kittens;
using namespace gfx1031_gemm;

namespace {
constexpr int S = 2;
} // namespace

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_one_wave_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_tile (&A_st)[S] = al.allocate<A_tile, S>();
    B_tile (&B_st)[S] = al.allocate<B_tile, S>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m   = blockIdx.x;
    const int tile_n   = blockIdx.y;
    const int wid      = warpid();
    const int warp_r   = wid / WARPS_N;
    const int warp_c   = wid % WARPS_N;
    const int k_blocks = K / BLOCK_K;
    if (k_blocks <= 0) return;

    const int warp_off_a = warp_r * WARP_M * K_STEP;
    const int warp_off_b = warp_c * WARP_N * K_STEP;

    auto issue_fill = [&](int slot, int kblock) {
        if (wid == 0) {
            rt_e<16, K_STEP> fill_rt;
            kittens::load(fill_rt, g.a, {0, 0, tile_m * (BLOCK_M / 16), kblock});
            kittens::sync::wait_load<0>();
            kittens::store(A_st[slot], fill_rt);
        }
        if (wid == 1) {
            rt_e<16, K_STEP> fill_rt;
            kittens::load(fill_rt, g.b, {0, 0, tile_n * (BLOCK_N / 16), kblock});
            kittens::sync::wait_load<0>();
            kittens::store(B_st[slot], fill_rt);
        }
    };

    issue_fill(0, 0);
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();

    rt_e<WARP_M, K_STEP> A_reg[2];
    rt_e<WARP_N, K_STEP> B_reg[2];

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S, nxt = (kb + 1) % S;
        const int fk = (kb + 1 < k_blocks) ? (kb + 1) : (k_blocks - 1);
        if (kb + 1 < k_blocks) issue_fill(nxt, fk);

        for (int si = 0; si < K_SUBBLOCKS - 1; ++si) {
            const int c = si & 1, n = 1 - c;
            kittens::load(A_reg[n], A_st[cur], warp_off_a + (si + 1) * K_STEP);
            kittens::load(B_reg[n], B_st[cur], warp_off_b + (si + 1) * K_STEP);
            kittens::sync::wait_ds<0>();
            mma_ABt(C_acc, A_reg[c], B_reg[c], C_acc);
        }

        constexpr int c_last = (K_SUBBLOCKS - 1) & 1;
        kittens::sync::wait_ds<0>();
        kittens::sync::sync();
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);
        kittens::sync::sync();
    }

    kittens::sync::wait_ds<0>();
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    const size_t mem_size = launch.dynamic_shared_memory<S>();

    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "11_gemm_one_wave: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1031_gemm::require_k_blocks(g.K(), "11_gemm_one_wave");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_one_wave_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_one_wave_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
