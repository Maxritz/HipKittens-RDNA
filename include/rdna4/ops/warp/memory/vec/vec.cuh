/**
 * @file
 * @brief Vector-level memory operations (RDNA4 gfx1201).
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

template<ducks::rv::all RV>
__device__ inline void load(RV &dst, const buffer_resource& br, uint32_t offset) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    const char* src_ptr = base + offset;
    char* dst_ptr = reinterpret_cast<char*>(&dst.data[0]);
    const int tid = kittens::laneid();
    constexpr int total_elems = RV::rows * RV::cols;
    constexpr int elems_per_load = 16 / sizeof(typename RV::dtype);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += kittens::WARP_THREADS * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename RV::dtype));
            *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename RV::dtype)) = val;
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

template<ducks::rv::all RV>
__device__ inline void store(const buffer_resource& br, const RV &src, uint32_t offset) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    char* dst_ptr = const_cast<char*>(base + offset);
    const char* src_ptr = reinterpret_cast<const char*>(&src.data[0]);
    const int tid = kittens::laneid();
    constexpr int total_elems = RV::rows * RV::cols;
    constexpr int elems_per_load = 16 / sizeof(typename RV::dtype);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += kittens::WARP_THREADS * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename RV::dtype));
            *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename RV::dtype)) = val;
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

} // namespace kittens