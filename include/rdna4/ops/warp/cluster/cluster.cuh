/**
 * @file
 * @brief Cluster-level operations for RDNA4 (gfx1201).
 *
 * RDNA4 supports workgroup clusters via __cluster_dims__. The cluster barrier
 * is implemented via s_barrier_signal/wait on the cluster's barrier ID.
 * On RDNA4, cluster operations map to the same s_barrier primitives as block-wide
 * operations, since the hardware manages cluster membership.
 */

#pragma once

#include <hip/hip_runtime.h>

namespace kittens {
namespace cluster {

struct cluster_group {
    static constexpr int threads = 32;
    static constexpr int rows = 16;
    static constexpr int cols = 16;
};

/**
 * @brief Signal a cluster-wide barrier.
 */
__device__ __forceinline__ void arrive() {
    asm volatile("s_barrier_signal -3" ::: "memory");
}

/**
 * @brief Wait on a cluster-wide barrier.
 */
__device__ __forceinline__ void wait() {
    asm volatile("s_barrier_wait -3" ::: "memory");
}

/**
 * @brief Fused cluster barrier (signal + wait).
 */
__device__ __forceinline__ void sync() {
    arrive();
    wait();
}

/**
 * @brief Sync all workgroups across the cluster.
 */
__device__ __forceinline__ void sync_all() {
    sync();
}

/**
 * @brief Async copy within a cluster via group LDS.
 * On RDNA4, cluster-wide copy is implemented as a synchronous load/store pair.
 */
template<typename tile_type>
__device__ inline void cp_async(tile_type &dst, const void* src) {
    const char* s = static_cast<const char*>(src);
    char* d = reinterpret_cast<char*>(&dst.data[0]);
    #pragma unroll
    for (int i = 0; i < sizeof(tile_type); i += 16) {
        if (i + 16 <= sizeof(tile_type)) {
            *reinterpret_cast<float4*>(d + i) = *reinterpret_cast<const float4*>(s + i);
        }
    }
}

} // namespace cluster
} // namespace kittens