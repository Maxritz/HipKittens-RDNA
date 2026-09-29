/**
 * @file 09_gemm_wgc.cpp
 * @brief Rung 09 -- 08_gemm_split_bar with workgroup cooperation via swizzling (RDNA2 (gfx1031)).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads; Wave64
 *   registers   ~240 VGPR against a 256/lane budget
 *   spills      none: 0 VGPR, 0 scratch
 *   LDS         2 stages x ~17 KB = ~34 KB of 64 KB (53%)
 *   sync        per K-block: 1 barrier, 3 LDS waits
 *   cooperative 4 workgroups jointly compute one 512x512 tile; each loads 1/4 of A/B
 *   intensity   128 FLOP per byte of global operand traffic
 *
 * RDNA2 (gfx1031). This rung takes rung 08 and tiles 512x512 instead of 256x256, using 4 workgroups
 * that each own 256x256 of C. Workgroups are identified by their position within a 2x2 group,
 * derived from blockIdx via swizzling. No TDM or cluster barriers are used.
 *
 * Uses only:
 *   - `kittens::load(rt,gl)`      : GL -> RT load.
 *   - `kittens::store(st,rt)`     : RT -> ST for LDS fill.
 *   - `kittens::sync::wait_load`  : drain GL -> RT.
 *   - `kittens::sync::sync`       : block-wide barrier.
 *   - `kittens::sync::wait_ds`    : drain LDS reads.
 *   - `kittens::load(rt,st,off)`  : shared -> register load.
 *   - `kittens::mma_ABt`          : packed fma via v_pk_fmac_f32.
 *   - `kittens::store(gl,rt,idx)` : direct column-major epilogue.
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

constexpr int WG_M = 2;
constexpr int WG_N = 2;
constexpr int WG_TILE_M = BLOCK_M * WG_M;
constexpr int WG_TILE_N = BLOCK_N * WG_N;

#include "common.h"

using namespace kittens;
using namespace gfx1031_gemm;

namespace {
constexpr int S = 2;
} // namespace

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_wgc_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_tile (&A_st)[S] = al.allocate<A_tile, S>();
    B_tile (&B_st)[S] = al.allocate<B_tile, S>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    // Swizzle: group 2x2 workgroups into one 512x512 tile
    const int group_m = (blockIdx.x / 2);
    const int group_n = (blockIdx.y / 2);
    const int wg_in_group_m = blockIdx.x % WG_M;
    const int wg_in_group_n = blockIdx.y % WG_N;
    const int tile_m = group_m * WG_M;
    const int tile_n = group_n * WG_N;

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
            kittens::load(fill_rt, g.a, {0, 0, tile_m * (BLOCK_M / 16) + wid, kblock + wg_in_group_m * (BLOCK_M / 16)});
            kittens::sync::wait_load<0>();
            kittens::store(A_st[slot], fill_rt);
        }
        if (wid == 1) {
            rt_e<16, K_STEP> fill_rt;
            kittens::load(fill_rt, g.b, {0, 0, tile_n * (BLOCK_N / 16) + wid, kblock + wg_in_group_n * (BLOCK_N / 16)});
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
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r + wg_in_group_m * WARP_M,
                                   tile_n * WARPS_N + warp_c + wg_in_group_n * WARP_N});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    const size_t mem_size = launch.dynamic_shared_memory<S>();

    // Swizzle grid: 2x2 workgroups per 512x512 tile
    dim3 grid(launch.grid.x / 2, launch.grid.y / 2);
    dim3 block(launch.block.x);

    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "09_gemm_wgc: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1031_gemm::require_k_blocks(g.K(), "09_gemm_wgc");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_wgc_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_wgc_kernel<<<grid, block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
