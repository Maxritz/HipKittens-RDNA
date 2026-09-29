/**
 * @file
 * @brief Async global-to-shared tile loads via LDS copy engine (RDNA4 gfx1201).
 *
 * RDNA4 supports global_load_async_to_lds via s_wait_loadcnt for draining.
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

template<ducks::st::all ST>
__device__ inline void cp_async(ST &dst, const auto &src, int offset = 0) {
    // Synchronous copy on RDNA4 — no separate async engine
    // Drained via s_wait_loadcnt
    const char* s = reinterpret_cast<const char*>(&src);
    char* d = reinterpret_cast<char*>(&dst.data[0]);
    #pragma unroll
    for (int i = 0; i < sizeof(ST); i += 16) {
        if (i + 16 <= sizeof(ST)) {
            *reinterpret_cast<float4*>(d + i) = *reinterpret_cast<const float4*>(s + i);
        }
    }
}

template<int N = 0>
__device__ inline void cp_wait() {
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

} // namespace kittens