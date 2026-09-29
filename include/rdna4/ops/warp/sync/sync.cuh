/**
 * @file
 * @brief Warp-level synchronization primitives for RDNA4 (gfx1201).
 *
 * RDNA4 has split counters like CDNA5:
 * - s_wait_vmcnt: wait for vector memory (global loads, VMEM)
 * - s_wait_dscnt: wait for DS (LDS) operations
 * - s_wait_kmcnt: wait for data fetch (K-cache / constant)
 * - s_wait_loadcnt: wait for async loads
 *
 * RDNA4 has no named barriers (NUM_NAMED_BARRIERS=0).
 */

#pragma once

#include <hip/hip_runtime.h>
#include "barrier.cuh"

namespace kittens {

namespace barrier {

struct named {
    static constexpr int num = 0; // No named barriers on RDNA4
};

}

namespace sync {

/**
 * @brief Wait for outstanding global loads, leaving up to N in flight.
 */
template<int N = 0>
__device__ __forceinline__ void wait_load() {
    static_assert(N >= 0 && N < 64, "loadcnt is 6-bit; max 63");
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding async copies / TDM.
 */
template<int N = 0>
__device__ __forceinline__ void wait_tdm() {
    static_assert(N >= 0 && N < 64, "loadcnt is 6-bit; max 63");
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding global stores, leaving up to N in flight.
 */
template<int N = 0>
__device__ __forceinline__ void wait_store() {
    static_assert(N >= 0 && N < 64, "storecnt is 6-bit; max 63");
    asm volatile("s_wait_storecnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding LDS reads, leaving up to N in flight.
 */
template<int N = 0>
__device__ __forceinline__ void wait_ds() {
    static_assert(N >= 0 && N < 64, "dscnt is 6-bit; max 63");
    asm volatile("s_wait_dscnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding data fetches (constant/K-cache), leaving up to N.
 */
template<int N = 0>
__device__ __forceinline__ void wait_km() {
    static_assert(N >= 0 && N < 64, "kmcnt is 6-bit; max 63");
    asm volatile("s_wait_kmcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for async copy operations.
 * RDNA4 supports s_wait_asynccnt for async copy engine drains.
 */
template<int N = 0>
__device__ __forceinline__ void wait_async() {
    static_assert(N >= 0 && N < 64, "asynccnt is 6-bit; max 63");
    asm volatile("s_wait_asynccnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Block-wide barrier (signal + wait).
 */
__device__ __forceinline__ void sync() {
    s_barrier();
}

/**
 * @brief Signal a block-wide split barrier.
 */
__device__ __forceinline__ void arrive() {
    asm volatile("s_barrier_signal -1" ::: "memory");
}

/**
 * @brief Wait on a block-wide split barrier.
 */
__device__ __forceinline__ void wait() {
    asm volatile("s_barrier_wait -1" ::: "memory");
}

/**
 * @brief Sync all workgroups (no cluster on RDNA4 — maps to block-wide sync).
 */
__device__ __forceinline__ void sync_all() {
    sync();
}

} // namespace sync

namespace sched {

/**
 * @brief Compiler fence to prevent instruction reordering across a boundary.
 */
__device__ __forceinline__ void compiler_fence() {
    __builtin_amdgcn_s_memtime_fence();
}

/**
 * @brief Pin SIMD priority using s_setprio.
 */
__device__ __forceinline__ void set_priority(int prio) {
    asm volatile("s_setprio %0" :: "s"(prio) : "memory");
}

} // namespace sched
} // namespace kittens