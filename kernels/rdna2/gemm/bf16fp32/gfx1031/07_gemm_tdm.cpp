/**
 * @file 07_gemm_segment.cpp
 * @brief Rung 07 -- 06_gemm_segment with a producer-consumer fill scheme (RDNA2 (gfx1031)).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads; Wave64
 *   registers   ~220 VGPR against a 256/lane budget; ~128 are accumulator
 *   spills      none: 0 VGPR, 0 scratch
 *   LDS         2 stages x ~17 KB = ~34 KB of 64 KB (53%)
 *   sync        per K-block: 1 barrier (full), 4 LDS waits (3 partial, 1 full)
 *   producer    two issuer warps per operand fill the next LDS stage as compute drains the current
 *   intensity   128 FLOP per byte of global operand traffic
 *
 * RDNA2 (gfx1031). This rung replaces CDNA5's `tdm::load_async` with a producer-consumer scheme:
 * two issuer warps (wid 0 and wid 1) use `kittens::load` + `kittens::store` to fill the next LDS
 * stage while consumer warps compute on the current one. The `wait_load` + `wait_ds` pattern
 * provides the same dependency ordering that `wait_tdm` gave on CDNA5.
 *
 * Uses only:
 *   - `kittens::load(rt,gl)`      : GL -> RT load for operand fetch.
 *   - `kittens::store(st,rt)`     : RT -> ST for LDS fill.
 *   - `kittens::sync::wait_load`  : drain GL -> RT before consuming.
 *   - `kittens::sync::sync`       : block-wide barrier.
 *   - `kittens::sync::wait_ds`    : drain LDS reads before stage handoff.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide ds_load_b128).
 *   - `kittens::mma_ABt`          : packed fma via v_pk_fmac_f32.
 *   - `kittens::sched::compiler_fence` : hold ops inside the barrier window.
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
using namespace gfx1031_gemm;

namespace {
constexpr int S = 2;
constexpr int FILL_ROWS = 16;
using fill_rt_e = rt_e<FILL_ROWS, K_STEP>;
using fill_st_e = st_e<FILL_ROWS, K_STEP>;
} // namespace

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_segment_kernel(const gemm_globals g, int M, int N, int K)
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

    constexpr int DS_SUB = ds_loads_per_subblock<WARP_M>()
                         + ds_loads_per_subblock<WARP_N>();

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
            kittens::sched::compiler_fence();
            kittens::load(A_reg[n], A_st[cur], warp_off_a + (si + 1) * K_STEP);
            kittens::load(B_reg[n], B_st[cur], warp_off_b + (si + 1) * K_STEP);
            kittens::sched::compiler_fence();
            kittens::sync::wait_ds<DS_SUB>();
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
            "07_gemm_segment: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1031_gemm::require_k_blocks(g.K(), "07_gemm_segment");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_segment_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_segment_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
