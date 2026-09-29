/**
 * @file
 * @brief Global-to-shared tile loads (RDNA4 gfx1201).
 *
 * RDNA4 has no TDM (Tile DMA) engine or cp.async, so all loads are synchronous.
 * The API mirrors CDNA5's signatures for source compatibility:
 * - load(ST& dst, const GL& src, const COORD& idx)
 * - load<NUM_THREADS>(st<T,ROWS,COLS,Shape>& dst, const GL& src, const COORD& idx, int row_stride)
 * - load_async(...)  -- synchronous on RDNA4
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// Buffer resource load/store for RDNA4
template<ducks::st::all ST>
__device__ inline void load(ST &dst, const buffer_resource& br, uint32_t offset) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    const char* src_ptr = base + offset;
    char* dst_ptr = reinterpret_cast<char*>(&dst.data[0]);
    const int laneid = kittens::laneid();
    constexpr int total_elems = ST::rows * ST::cols;
    constexpr int elems_per_load = 16 / sizeof(typename ST::dtype);
    constexpr int total_calls = (total_elems + elems_per_load - 1) / elems_per_load;

    #pragma unroll
    for (int i = laneid * elems_per_load; i < total_elems;
         i += kittens::WARP_THREADS * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename ST::dtype));
            *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename ST::dtype)) = val;
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

template<ducks::st::all ST>
__device__ inline void store(const buffer_resource& br, const ST &src, uint32_t offset) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    char* dst_ptr = const_cast<char*>(base + offset);
    const char* src_ptr = reinterpret_cast<const char*>(&src.data[0]);
    const int laneid = kittens::laneid();
    constexpr int total_elems = ST::rows * ST::cols;
    constexpr int elems_per_load = 16 / sizeof(typename ST::dtype);
    constexpr int total_calls = (total_elems + elems_per_load - 1) / elems_per_load;

    #pragma unroll
    for (int i = laneid * elems_per_load; i < total_elems;
         i += kittens::WARP_THREADS * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename ST::dtype));
            *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename ST::dtype)) = val;
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

/**
 * @brief Load from GL to ST with axis-based stride (RDNA2-style, for source compat).
 */
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
 * Source-compatible with CDNA5's `load<NUM_THREADS>(dst, src, idx, row_stride)`.
 * On RDNA4, the load is synchronous (no async-to-LDS engine).
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
        float4 val = load_global_vec4((const float4*)g_ptr);
        uint2 packed = {reinterpret_cast<uint2*>(&val)[0], reinterpret_cast<uint2*>(&val)[1]};
        store_shared_vec(dst.idx(dst_ptr, {row, col}), packed);
    }
    #ifdef BUILTINS_ONLY
    __builtin_amdgcn_s_waitcnt(0);
    #else
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    #endif
}

/**
 * @brief Async-style load (synchronous on RDNA4, no TDM/cp.async engine).
 */
template<int N_THREADS = WARP_THREADS, bool RATE_ONLY = false, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL,
         ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    load<N_THREADS>(dst, src, idx, row_stride);
}

} // namespace kittens