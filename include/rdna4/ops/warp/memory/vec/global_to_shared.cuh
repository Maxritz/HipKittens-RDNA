/**
 * @file
 * @brief Global-to-shared vector loads (RDNA4 gfx1201).
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

template<ducks::sv::all SV>
__device__ inline void load(SV &dst, const buffer_resource& br, uint32_t offset, bool swap_lanes) {
    const char* base = reinterpret_cast<const char*>(br.ptr);
    const char* src_ptr = base + offset;
    char* dst_ptr = reinterpret_cast<char*>(&dst.data[0]);
    const int tid = kittens::laneid();
    constexpr int total_elems = SV::rows * SV::cols;
    constexpr int elems_per_load = 16 / sizeof(typename SV::dtype);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += kittens::WARP_THREADS * elems_per_load) {
        if (i + elems_per_load <= total_elems) {
            float4 val = *reinterpret_cast<const float4*>(src_ptr + i * sizeof(typename SV::dtype));
            if (swap_lanes) {
                int grp = i / 4;
                int off = i % 4;
                int swapped = grp * 4 + (off < 2 ? off + 2 : off - 2);
                if (swapped + elems_per_load <= total_elems) {
                    float4 swval;
                    float2* f2 = reinterpret_cast<float2*>(&val);
                    float2* sf2 = reinterpret_cast<float2*>(&swval);
                    if (off < 2) { sf2[0] = f2[1]; sf2[1] = f2[0]; }
                    else { sf2[0] = f2[1]; sf2[1] = f2[0]; }
                    *reinterpret_cast<float4*>(dst_ptr + swapped * sizeof(typename SV::dtype)) = swval;
                }
            } else {
                *reinterpret_cast<float4*>(dst_ptr + i * sizeof(typename SV::dtype)) = val;
            }
        }
    }
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

} // namespace kittens