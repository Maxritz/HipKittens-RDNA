/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template< int  axis, bool assume_aligned,
          ducks::st::all ST, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<ST>,
          int  N_THREADS = WARP_THREADS >
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    const int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4    buf[small_calls];

    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                buf[j] = load_global_vec4_async((float4*) (src_ptr + (row * row_stride + col)));
            }
        }

        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt vmcnt(0)");
        #endif

        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf[j].x, buf[j].y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf[j].z, buf[j].w});
            }
        }

        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
    } 
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

/**
 * @brief Load a shared tile from global memory with a specified row stride.
 *
 * This overload mirrors the CDNA5 `load<NUM_THREADS>(dst, src, idx, row_stride)` signature
 * for source compatibility. On RDNA2 (no async-to-LDS engine), the load is synchronous:
 * each thread loads 16 bytes, then s_waitcnt drains before the caller proceeds.
 *
 * @tparam N_THREADS    Number of threads participating in the load.
 * @param  dst          Destination st (shared) tile.
 * @param  src          Global tile descriptor.
 * @param  idx          Tile coordinate inside src.
 * @param  row_stride   Element stride between rows in src.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL,
         ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                            const COORD& idx, int row_stride)
{
    constexpr int elems_per_load = 16 / sizeof(T);
    constexpr int total_elems    = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += N_THREADS * elems_per_load)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const T* g_ptr = base + (row * row_stride + col);
        // Synchronous load: each lane loads 16 bytes
        float4 val = load_global_vec4((const float4*)g_ptr);
        uint2 packed = {reinterpret_cast<uint2*>(&val)[0], reinterpret_cast<uint2*>(&val)[1]};
        store_shared_vec(dst.idx(dst_ptr, {row, col}), packed);
    }
    // Drain before caller proceeds
    #ifdef BUILTINS_ONLY
    __builtin_amdgcn_s_waitcnt(0);
    #else
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    #endif
}


/********************************************* Register Pipelining **********************************************/

/**
 * @brief Load from global memory into a register tile, with cooperative pipelining.
 *
 * On RDNA2 (no cp.async), this is a simple synchronous load. The `wait_async` 
 * sync primitive is a no-op that calls __syncthreads to maintain API compatibility.
 */

template<int N_THREADS = WARP_THREADS, bool RATE_ONLY = false, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    // RDNA2 has no async copy engine — load synchronously
    load<N_THREADS>(dst, src, idx, row_stride);
}

} // namespace kittens