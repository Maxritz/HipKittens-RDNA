/**
 * @file
 * @brief Warp-level synchronization primitives for RDNA2 (gfx1031).
 *
 * RDNA2 has no named barriers and no split counters (no s_wait_loadcnt, s_wait_dscnt, etc.).
 * Synchronization is done via:
 * - s_barrier: block-wide barrier
 * - s_waitcnt: unified wait for VM, LDS, and other ops
 * - No TDM (Tile DMA) engine or async copy engine
 */

#pragma once

#include <hip/hip_runtime.h>
#include "barrier.cuh"

namespace kittens {
namespace sync {

/**
 * @brief Wait for outstanding global loads, leaving up to N in flight.
 * On RDNA2, there is no s_wait_loadcnt — global loads use the VM counter.
 * This maps to s_waitcnt vmcnt(N).
 */
template<int N = 0>
__device__ __forceinline__ void wait_load() {
    unsigned int mask = N & 0xFF;
    asm volatile("s_waitcnt vmcnt(%0)" :: "i"(mask) : "memory");
}

/**
 * @brief Wait for outstanding async copies / TDM.
 * On RDNA2 there is no TDM engine, so this is a no-op that maps to wait_load.
 * The producer-consumer pattern in rung 07 replaces TDM with explicit load/store.
 */
template<int N = 0>
__device__ __forceinline__ void wait_tdm() {
    wait_load<N>();
}

/**
 * @brief Wait for outstanding stores.
 * On RDNA2, stores complete via the VM counter.
 */
template<int N = 0>
__device__ __forceinline__ void wait_store() {
    unsigned int mask = N & 0xFF;
    asm volatile("s_waitcnt vmcnt(%0)" :: "i"(mask) : "memory");
}

/**
 * @brief Wait for outstanding LDS reads, leaving up to N in flight.
 * On RDNA2, there is no s_wait_dscnt — LDS uses the lgkmcnt counter in s_waitcnt.
 */
template<int N = 0>
__device__ __forceinline__ void wait_ds() {
    unsigned int mask = N & 0xFF;
    asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(mask) : "memory");
}

/**
 * @brief Wait for outstanding data fetches (constant/K-cache).
 * On RDNA2, there is no s_wait_kmcnt — uses the lgkmcnt counter.
 */
template<int N = 0>
__device__ __forceinline__ void wait_km() {
    unsigned int mask = N & 0xFF;
    asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(mask) : "memory");
}

/**
 * @brief Wait for async copy operations (no-op on RDNA2, since no async copy engine).
 */
template<int N = 0>
__device__ __forceinline__ void wait_async() {
    __syncthreads();
}

/**
 * @brief Block-wide barrier (signal + wait).
 * Equivalent to __syncthreads() but without the implicit memory drain.
 */
__device__ __forceinline__ void sync() {
    asm volatile("s_barrier" ::: "memory");
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
 * @brief Sync all workgroups (not available on RDNA2 — maps to block-wide sync).
 * On RDNA2 with no cluster, this is equivalent to sync().
 */
__device__ __forceinline__ void sync_all() {
    sync();
}

} // namespace sync

namespace sched {

/**
 * @brief Compiler fence to prevent instruction reordering across a boundary.
 * On RDNA2, s_sched_barrier is not available — use a memory fence instead.
 */
__device__ __forceinline__ void compiler_fence() {
    __builtin_amdgcn_s_memtime_fence();
}

/**
 * @brief Pin SIMD priority using s_setprio.
 * RDNA2 supports s_setprio for QoS pinning.
 */
__device__ __forceinline__ void set_priority(int prio) {
    asm volatile("s_setprio %0" :: "s"(prio) : "memory");
}

} // namespace sched
} // namespace kittens