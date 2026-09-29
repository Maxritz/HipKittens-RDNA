/**
 * @file
 * @brief Block-wide barrier primitives for RDNA2 (gfx1031).
 *
 * RDNA2 uses s_barrier / s_barrier_signal / s_barrier_wait for block-wide
 * synchronization. There are no named barriers on RDNA2.
 */

#pragma once

#include <hip/hip_runtime.h>

namespace kittens {

namespace sync {

/**
 * @brief Block-wide barrier using s_barrier.
 */
__device__ __forceinline__ void s_barrier() {
    asm volatile("s_barrier" ::: "memory");
}

} // namespace sync

} // namespace kittens