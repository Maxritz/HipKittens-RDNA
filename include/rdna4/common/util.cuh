/**
 * @file
 * @brief General utilities for ThunderKittens on RDNA4 (gfx1201).
 *
 * RDNA4 constants differ from CDNA5:
 * - 64 LDS banks (vs CDNA5's 32+32)
 * - 256 KB LDS per CU (vs CDNA5's 320 KB / segment-aligned)
 * - No NUMA segment split (that is a CDNA2/CDNA5 multi-chiplet feature)
 * - wave-32 (same as CDNA5)
 */

#pragma once

#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

#include <hip/hip_runtime.h>

#include "base_types.cuh"

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif

namespace kittens {

constexpr int WARP_THREADS{32};

__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; }
__device__ __forceinline__ int num_warps() { return blockDim.x / WARP_THREADS; }
__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }

using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
using i32x3 = int32_t __attribute__((ext_vector_type(3)));
using u32x3 = uint32_t __attribute__((ext_vector_type(3)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

constexpr int MAX_SHARED_MEMORY = 256 * 1024; // 256 KB LDS on RDNA4
constexpr int NUM_XCDS = 1;
constexpr int CUS_PER_XCD = 1;
constexpr int NUM_CUS = 1; // single-XCD partion, CU count varies by SKU

#ifndef KITTENS_NUM_NUMA
#define KITTENS_NUM_NUMA 1
#endif
constexpr int NUM_NUMA = KITTENS_NUM_NUMA;

namespace ducks {
struct default_type {};
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type
}

static constexpr uint64_t MASK_ALL = 0xFFFFFFFFFFFFFFFF;

template<typename T>
__device__ static inline T packed_shfl_down(uint64_t mask, const T &f, int delta) {
    if constexpr (std::is_same_v<T, bf16_2> || std::is_same_v<T, bf16>) {
        static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
        union {
           __hip_bfloat162 bf162;
           unsigned int ui;
        } u;
        if constexpr (std::is_same_v<T, bf16_2>) {
            u.bf162 = *reinterpret_cast<const __hip_bfloat162*>(&f);
        } else {
            u.bf162 = __hip_bfloat162{*reinterpret_cast<const __hip_bfloat16*>(&f),
                                       *reinterpret_cast<const __hip_bfloat16*>(&f)};
        }
        u.ui = __shfl_down_sync<unsigned long long, unsigned int>(mask, u.ui, delta, 64);
        if constexpr (std::is_same_v<T, bf16>) {
            return *reinterpret_cast<const T*>(&u.bf162.x);
        } else {
            return u.bf162;
        }
    } else {
        return __shfl_down(f, delta);
    }
}
template<>
__device__ inline float2 packed_shfl_down<float2>(uint64_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down(f.x, delta);
    r.y = __shfl_down(f.y, delta);
    return r;
}
template<typename T>
__device__ static inline T packed_shfl(uint64_t mask, const T &f, int src) {
    return __shfl(f, src);
}
template<>
__device__ inline bf16 packed_shfl(uint64_t mask, const bf16 &f, int src) {
    float r = __shfl(base_types::convertor<float, bf16>::convert(f), src);
    return base_types::convertor<bf16, float>::convert(r);
}
template<>
__device__ inline bf16_2 packed_shfl(uint64_t mask, const bf16_2 &f, int src) {
    float2 r;
    r.x = __shfl(base_types::convertor<float, bf16>::convert(f.x), src);
    r.y = __shfl(base_types::convertor<float, bf16>::convert(f.y), src);
    return base_types::convertor<bf16_2, float2>::convert(r);
}
template<>
__device__ inline half packed_shfl(uint64_t mask, const half &f, int src) {
    float r = __shfl(base_types::convertor<float, half>::convert(f), src);
    return base_types::convertor<half, float>::convert(r);
}
template<>
__device__ inline half_2 packed_shfl(uint64_t mask, const half_2 &f, int src) {
    float2 r;
    r.x = __shfl(base_types::convertor<float, half>::convert(f.x), src);
    r.y = __shfl(base_types::convertor<float, half>::convert(f.y), src);
    return base_types::convertor<half_2, float2>::convert(r);
}
template<>
__device__ inline float2 packed_shfl<float2>(uint64_t mask, const float2 &f, int src) {
    float2 r;
    r.x = __shfl(f.x, src);
    r.y = __shfl(f.y, src);
    return r;
}

using bytes_4  = HIP_vector_type<float, 1>;
using bytes_8  = HIP_vector_type<float, 2>;
using bytes_16 = HIP_vector_type<float, 4>;

#define KITTENS_ALIGN_AS(n) alignas(n)
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)

struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };

template<int default_alignment=16>
struct shared_allocator {
    int *ptr;
    __device__ shared_allocator(int *_ptr): ptr(_ptr) {}
    template<typename A>
    __device__ inline A& allocate() {
        if constexpr (default_alignment > 0) {
            const uint64_t misalign = reinterpret_cast<uint64_t>(ptr) % default_alignment;
            if (misalign != 0) ptr += (default_alignment - misalign) / sizeof(int);
        }
        A*p = reinterpret_cast<A*>(ptr);
        ptr += sizeof(A)/sizeof(int);
        return *p;
    }
};

} // namespace kittens
