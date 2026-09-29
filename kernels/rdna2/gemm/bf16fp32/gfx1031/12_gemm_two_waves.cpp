/**
 * @file 12_gemm_two_waves.cpp
 * @brief Rung 12 -- 11_gemm_one_wave with two waves sharing a 256x tile (RDNA2 (gfx1031)).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x128 macro, 64x64 per warp, 4x2 warps; BLOCK_K 32 = 1 x K_STEP 32
 *   occupancy   4 warps / 128 threads / 2 waves per SIMD (Wave64); up to 4 workgroups per CU
 *   registers   ~180 VGPR; 64 are accumulator (doubled for wider tile)
 *   spills      none: 0 VGPR, 0 scratch
 *   LDS         2 stages x ~8.5 KB = ~17 KB of 64 KB (27%)
 *   sync        per K-block: 1 barrier, 2 LDS drains
 *   intensity   32 FLOP per byte of global operand traffic
 *
 * RDNA2 (gfx1031). This rung takes rung 11's 128x128 wave tile, widens it to 256x128 (4x2 warps, 128
 * threads) to use two waves per SIMD. No TDM or cluster is used.
 */

#include "kittens.cuh"

constexpr int BLOCK_M     = 256;
constexpr int BLOCK_N     = 128;
constexpr int BLOCK_K     = 32;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 4;
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
void gemm_two_waves_kernel(const gemm_globals g, int M, int N, int K)
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
        if (wid < WARPS_M) {
            rt_e<16, K_STEP> fill_rt;
            kittens::load(fill_rt, g.a, {0, 0, tile_m * (BLOCK_M / 16) + wid, kblock});
            kittens::sync::wait_load<0>();
            kittens::store(*reinterpret_cast<A_tile::subtile*>(&A_st[slot].data[A_tile::idx(wid * 16, 0)]), fill_rt);
        } else {
            rt_e<16, K_STEP> fill_rt;
            kittens::load(fill_rt, g.b, {0, 0, tile_n * (BLOCK_N / 16) + (wid - WARPS_M), kblock});
            kittens::sync::wait_load<0>();
            kittens::store(*reinterpret_cast<B_tile::subtile*>(&B_st[slot].data[B_tile::idx((wid - WARPS_M) * 16, 0)]), fill_rt);
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
            "12_gemm_two_waves: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1031_gemm::require_k_blocks(g.K(), "12_gemm_two_waves");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_two_waves_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_two_waves_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
