/**
 * @file 10_gemm_epilogue.cpp
 * @brief Rung 10 -- 09_gemm_wgc plus an LDS-staged C epilogue (RDNA2 (gfx1031)).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads; Wave64
 *   registers   ~240 VGPR against a 256/lane budget
 *   spills      none: 0 VGPR, 0 scratch
 *   LDS         2 A/B stages + 1 C stage = ~34 KB of 64 KB (53%)
 *   sync        per K-block: 1 barrier, 2 LDS drains; per epilogue: 1 barrier, 1 drain
 *   intensity   128 FLOP per byte of global operand traffic
 *
 * RDNA2 (gfx1031). This rung takes rung 09 and adds an LDS-staged epilogue: C accumulates in RTs,
 * then gets stored through a shared C tile for coalesced column-major writes. No TDM is used.
 *
 * Uses only:
 *   - `kittens::load(rt,gl)`      : GL -> RT load.
 *   - `kittens::store(st,rt)`     : RT -> ST for LDS fill.
 *   - `kittens::sync::wait_load`  : drain GL -> RT.
 *   - `kittens::sync::sync`       : block-wide barrier.
 *   - `kittens::sync::wait_ds`    : drain LDS reads.
 *   - `kittens::load(rt,st,off)`  : shared -> register load.
 *   - `kittens::mma_ABt`          : packed fma via v_pk_fmac_f32.
 *   - `kittens::store(st,rt)`     : RT -> shared C tile.
 *   - `kittens::store(gl,st,idx)` : column-major epilogue from LDS.
 */

#include "kittens.cuh"

constexpr int BLOCK_M     = 256;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 4;
constexpr int WARPS_N     = 4;
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
void gemm_epilogue_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_tile (&A_st)[S] = al.allocate<A_tile, S>();
    B_tile (&B_st)[S] = al.allocate<B_tile, S>();
    C_tile C_st = *al.allocate<C_tile>();

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
    kittens::store(C_st, C_acc);
    kittens::sync::sync();
    kittens::store(g.c, C_st, {tile_m * WARPS_M, tile_n * WARPS_N});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    const size_t mem_size = S * (sizeof(A_tile) + sizeof(B_tile)) + sizeof(C_tile);

    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "10_gemm_epilogue: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1031_gemm::require_k_blocks(g.K(), "10_gemm_epilogue");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_epilogue_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_epilogue_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
