/**
 * @file
 * @brief Group-level vector memory operations (RDNA4 gfx1201).
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace group {
namespace sv {

template<ducks::sv::all SV>
__device__ inline void load(SV &dst, const buffer_resource& br, uint32_t offset) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    const char* src_ptr = base + offset;
    char* dst_ptr = reinterpret_cast<char*>(&dst.data[0]);
    const int tid = threadIdx.x;
    constexpr int total_elems = SV::rows * SV::cols;
    constexpr int elems_per_load = 16 / sizeof(typename SV::dtype);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += blockDim.x * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename SV::dtype));
            *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename SV::dtype)) = val;
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

template<ducks::sv::all SV>
__device__ inline void store(const buffer_resource& br, const SV &src, uint32_t offset) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    char* dst_ptr = const_cast<char*>(base + offset);
    const char* src_ptr = reinterpret_cast<const char*>(&src.data[0]);
    const int tid = threadIdx.x;
    constexpr int total_elems = SV::rows * SV::cols;
    constexpr int elems_per_load = 16 / sizeof(typename SV::dtype);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += blockDim.x * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename SV::dtype));
            *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename SV::dtype)) = val;
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

} // namespace sv
} // namespace group
} // namespace kittens