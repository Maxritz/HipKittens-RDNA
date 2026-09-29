/**
 * @file
 * @brief Memory operation utilities (RDNA4 gfx1201).
 *
 * Provides inline assembly intrinsics for global and shared memory access.
 * RDNA4 uses the same global_load_dwordx* instructions as RDNA2,
 * but with split counter wait semantics (s_wait_vmcnt / s_wait_lgkmcnt).
 */

#pragma once

#include <cstdint>
#include <cstring>

namespace kittens {
namespace mem {

__device__ __forceinline__ void fence() {
    asm volatile("s_wait_vmcnt 0" ::: "memory");
}

} // namespace mem

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, 0x110000);

    row_stride_bytes &= 0x3FFF;
    i32x4 result = {{static_cast<int32_t>(rsrc.ptr & 0xFFFFFFFF),
                     static_cast<int32_t>((rsrc.ptr >> 32) & 0xFFFFFFFF),
                     static_cast<int32_t>(rsrc.range | (row_stride_bytes << 16)),
                     static_cast<int32_t>(rsrc.config)}};
    return result;
}

__device__ inline float4 load_global_vec4(const float4* gptr) {
    float4 v;
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        "s_wait_vmcnt 0\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline float2 load_global_vec2(const float2* gptr) {
    float2 v;
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        "s_wait_vmcnt 0\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline float4 load_global_vec4_async(const float4* gptr) {
    float4 v;
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline float2 load_global_vec2_async(const float2* gptr) {
    float2 v;
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline void store_global_b128_async(void* gptr, __uint128_t value) {
    asm volatile(
        "global_store_dwordx4 %0, %1, off\n"
        :
        : "v"(gptr), "v"(value)
        : "memory"
    );
}

__device__ inline float2 load_shared_vec(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        "s_wait_lgkmcnt 0\n"
        : "=v"(result)
        : "v"(lds_off)
        : "memory"
    );
    return result;
}

__device__ inline float2 load_shared_vec_async(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        : "=v"(result)
        : "v"(lds_off)
        : "memory"
    );
    return result;
}

__device__ inline void store_shared_vec(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        "s_wait_lgkmcnt 0\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

__device__ inline void store_shared_vec_async(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

} // namespace kittens