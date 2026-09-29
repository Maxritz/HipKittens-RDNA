/**
 * @file
 * @brief Scheduling primitives for RDNA4 (gfx1201).
 *
 * RDNA4 has no sched_barrier (CDNA5 priority scheduling feature).
 * Scheduling on RDNA4 is handled via s_memtime_fence and compiler-level scheduling.
 */

#pragma once

#include <hip/hip_runtime.h>

namespace kittens {
namespace sched {

struct scheduler {
    // No sched_barrier on RDNA4
};

/**
 * @brief Compiler fence to prevent instruction reordering across a boundary.
 */
__device__ __forceinline__ void compiler_fence() {
    __builtin_amdgcn_s_memtime_fence();
}

/**
 * @brief Pin SIMD priority using s_setprio.
 * On RDNA4, s_setprio is available for QoS pinning.
 */
__device__ __forceinline__ void set_priority(int prio) {
    asm volatile("s_setprio %0" :: "s"(prio) : "memory");
}

} // namespace sched
} // namespace kittens