/**
 * @file 08_gemm_split_bar.cpp
 * @brief Rung 08 -- 07_gemm_segment with a split barrier over the last matmul (RDNA4 (gfx1201)).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads; Wave32
 *   registers   ~234 VGPR against a 256/lane budget; ~128 are accumulator
 *   spills      none: 0 VGPR, 0 scratch
 *   LDS         2 stages x ~17 KB = ~34 KB of 64 KB (53%)
 *   sync        per K-block: 1 split barrier, 4 LDS waits (3 partial, 1 full)
 *   window      the K-block's last mma_ABt issues between arrive() and wait()
 *   producer    two issuer warps per operand fill the next LDS stage
 *   intensity   128 FLOP per byte of global operand traffic
 *
 * RDNA4 (gfx1201). This rung splits the workgroup barrier: arrive() is issued before the last matrix
 * op, and wait() after. The split is legal because the op reads registers only -- a warp is done
 * with the LDS stage the moment its ds_loads drain, which is before it finishes computing.
 *
 * Uses only:
 *   - `kittens::load(rt,gl)`      : GL -> RT load.
 *   - `kittens::store(st,rt)`     : RT -> ST for LDS fill.
 *   - `kittens::sync::wait_load`  : drain GL -> RT.
 *   - `kittens::sync::arrive`     : workgroup barrier arrive (split barrier signal).
 *   - `kittens::sync::wait`      : workgroup barrier wait (split barrier wait).
 *   - `kittens::sync::wait_ds`    : drain LDS reads.
 *   - `kittens::load(rt,st,off)`  : shared -> register load.
 *   - `kittens::mma_ABt`          : packed fma via v_pk_fmac_f32.
 *   - `kittens::sched::compiler_fence` : hold ops inside the signal-to-wait window.
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

#include "common.h"

using namespace kittens;
using namespace gfx1201_gemm;

namespace {
constexpr int S = 2;
constexpr int FILL_ROWS = 16;
using fill_rt_e = rt_e<FILL_ROWS, K_STEP>;
using fill_st_e = st_e<FILL_ROWS, K_STEP>;
} // namespace

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_split_bar_kernel(const gemm_globals g, int M, int N, int K)
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

    constexpr int DS_SUB = ds_loads_per_subblock<WARPS_M>()
                         + ds_loads_per_subblock<WARPS_N>();

    auto issue_fill = [&](int slot, int kblock) {
        if (wid == 0) {
            fill_rt_e fill_rt;
            fill_st_e& A_fill_st = *reinterpret_cast<fill_st_e*>(
                &A_st[slot].data[A_tile::idx(wid * FILL_ROWS, 0)]);
            kittens::load(fill_rt, g.a, {0, 0, tile_m * (BLOCK_M / FILL_ROWS) + wid, kblock});
            kittens::sync::wait_load<0>();
            kittens::store(A_fill_st, fill_rt);
        }
        if (wid == 1) {
            fill_rt_e fill_rt;
            fill_st_e& B_fill_st = *reinterpret_cast<fill_st_e*>(
                &B_st[slot].data[B_tile::idx(wid * FILL_ROWS, 0)]);
            kittens::load(fill_rt, g.b, {0, 0, tile_n * (BLOCK_N / FILL_ROWS) + wid, kblock});
            kittens::sync::wait_load<0>();
            kittens::store(B_fill_st, fill_rt);
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
            kittens::sync::wait_ds<DS_SUB>();
            mma_ABt(C_acc, A_reg[c], B_reg[c], C_acc);
        }

        constexpr int c_last = (K_SUBBLOCKS - 1) & 1;
        kittens::sync::wait_ds<0>();
        kittens::sync::arrive();
        kittens::sched::compiler_fence();
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);
        kittens::sched::compiler_fence();
        kittens::sync::wait();
    }

    kittens::sync::wait_ds<0>();
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    const size_t mem_size = launch.dynamic_shared_memory<S>();

    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "08_gemm_split_bar: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1201_gemm::require_k_blocks(g.K(), "08_gemm_split_bar");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_split_bar_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_split_bar_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
