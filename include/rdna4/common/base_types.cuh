/**
 * @file
 * @brief Declarations, manipulations, and wrappers for basic types (RDNA4 gfx1201).
 *
 * Same type aliases as CDNA5 but FP8 is IEEE/OCP (HIP_FP8_TYPE_OCP=1) rather
 * than the CDNA3 FNUZ encoding. The `constants` and `convertor` namespaces
 * mirror CDNA5 exactly (the RDNA4 variants are identical — FP8 format
 * differences are handled at the HIP runtime level).
 */

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp4.h>
#include <hip/amd_detail/amd_hip_ocp_types.h>
#include <hip/hip_runtime.h>
#include <string>
#include <bit>

#ifndef HIP_FP8_TYPE_OCP
#define HIP_FP8_TYPE_OCP 1
#endif

typedef uint32_t __amd_fp8x4_storage_t;

namespace kittens {

using bf16 = __hip_bfloat16;
using half = __half;
using bf16_2 = __hip_bfloat162;
using half_2 = __half2;
using fp8e4m3 = __hip_fp8_e4m3;
using fp8e4m3_2 = __hip_fp8x2_e4m3;
using fp8e4m3_4 = __hip_fp8x4_e4m3;
using fp8e8m0 = __amd_scale_t;
using fp8e8m0_2 = __amd_fp8x2_storage_t;
using fp8e8m0_4 = __amd_fp8x4_storage_t;
using fp4e2m1   = __hip_fp4_e2m1;
using fp4e2m1_2 = __hip_fp4x2_e2m1;
using fp4e2m1_4 = __hip_fp4x4_e2m1;

namespace ducks {
namespace base_types {

template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2> || std::is_same_v<T, fp8e4m3_4>
    || std::is_same_v<T, fp4e2m1_4>;
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3>
    || std::is_same_v<T, fp4e2m1>;

} // namespace base_types
} // namespace ducks

namespace base_types {

template<typename T> struct constants {
    static __device__ inline constexpr T zero()      { return T{0}; }
    static __device__ inline constexpr T ones()       { return T{1}; }
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; }
    static __device__ inline constexpr T neg_infty() { return T{-INFINITY}; }
};
template<> struct constants<float2> {
    static __device__ inline constexpr float2 zero()      { return float2{0.f, 0.f}; }
    static __device__ inline constexpr float2 ones()       { return float2{1.f, 1.f}; }
    static __device__ inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __device__ inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};
template<> struct constants<bf16> {
    static __device__ inline constexpr bf16 zero()      { return std::bit_cast<bf16>(uint16_t(0x0000)); }
    static __device__ inline constexpr bf16 ones()       { return std::bit_cast<bf16>(uint16_t(0x3F80)); }
    static __device__ inline constexpr bf16 pos_infty() { return std::bit_cast<bf16>(uint16_t(0x7F80)); }
    static __device__ inline constexpr bf16 neg_infty() { return std::bit_cast<bf16>(uint16_t(0xFF80)); }
};
template<> struct constants<bf16_2> {
    static __device__ inline bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static __device__ inline bf16_2 ones()       { return bf16_2{constants<bf16>::ones(),       constants<bf16>::ones()};       }
    static __device__ inline bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __device__ inline bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template<> struct constants<half> {
    static __device__ inline constexpr half zero()      { return std::bit_cast<half>(uint16_t(0x0000)); }
    static __device__ inline constexpr half ones()       { return std::bit_cast<half>(uint16_t(0x3C00)); }
    static __device__ inline constexpr half pos_infty() { return std::bit_cast<half>(uint16_t(0x7C00)); }
    static __device__ inline constexpr half neg_infty() { return std::bit_cast<half>(uint16_t(0xFC00)); }
};
template<> struct constants<half_2> {
    static __device__ inline constexpr half_2 zero()      { return std::bit_cast<half_2>(uint32_t(0x00000000)); }
    static __device__ inline constexpr half_2 ones()       { return std::bit_cast<half_2>(uint32_t(0x3C003C00)); }
    static __device__ inline constexpr half_2 pos_infty() { return std::bit_cast<half_2>(uint32_t(0x7C007C00)); }
    static __device__ inline constexpr half_2 neg_infty() { return std::bit_cast<half_2>(uint32_t(0xFC00FC00)); }
};
template<> struct constants<fp8e4m3> {
    static __device__ inline constexpr fp8e4m3 zero() { return std::bit_cast<fp8e4m3>(uint8_t(0x00)); }
    static __device__ inline constexpr fp8e4m3 one() { return std::bit_cast<fp8e4m3>(uint8_t(0x38)); }
};
template<> struct constants<fp8e4m3_2> {
    static __device__ inline constexpr fp8e4m3_2 zero() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp8e4m3_2 one() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x3838)); }
};
template<> struct constants<fp8e4m3_4> {
    static __device__ inline constexpr fp8e4m3_4 zero() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr fp8e4m3_4 one() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x38383838)); }
};
template<> struct constants<int> {
    static __device__ inline constexpr int zero()      { return 0; }
    static __device__ inline constexpr int ones()       { return 1; }
};
template<> struct constants<int2> {
    static __device__ inline constexpr int2 zero()      { return int2{0, 0}; }
    static __device__ inline constexpr int2 ones()       { return int2{1, 1}; }
};

template<typename T> struct packing {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = T;
    using packed_type = T;
    static __device__ inline constexpr T pack(const auto &i);
};
template<> struct packing<bf16> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<half> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<half_2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<float> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<float2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<int> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; }
};
template<> struct packing<int2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; }
};
template<> struct packing<float4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<fp8e4m3> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e4m3_4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e8m0> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e8m0;
    using packed_type = fp8e8m0_4;
};
template<> struct packing<fp8e8m0_4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e8m0;
    using packed_type = fp8e8m0_4;
};
template<> struct packing<fp4e2m1> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp4e2m1;
    using packed_type = fp4e2m1_4;
};
template<> struct packing<fp4e2m1_4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp4e2m1;
    using packed_type = fp4e2m1_4;
};

template<typename T, typename U> struct convertor {
    static __host__ __device__ inline T convert(const U & u) {
        return (T)u;
    }
};
template<> struct convertor<float, bf16> {
    static __host__ __device__ inline float convert(const bf16 & u) {
        return __bfloat162float(u);
    }
};
template<> struct convertor<bf16, float> {
    static __host__ __device__ inline bf16 convert(const float & u) {
        return __float2bfloat16(u);
    }
};
template<> struct convertor<float2, bf16_2> {
    static __host__ __device__ inline float2 convert(const bf16_2 & u) {
        return __bfloat1622float2(u);
    }
};
template<> struct convertor<bf16_2, float2> {
    static __host__ __device__ inline bf16_2 convert(const float2 &u) {
        uint32_t result;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                     : "=v"(result)
                     : "v"(u.x), "v"(u.y));
        return *reinterpret_cast<bf16_2*>(&result);
    }
};
template<> struct convertor<float, half> {
    static __host__ __device__ inline float convert(const half & u) {
        return __half2float(u);
    }
};
template<> struct convertor<half, float> {
    static __host__ __device__ inline half convert(const float & u) {
        return __float2half(u);
    }
};
template<> struct convertor<float2, half_2> {
    static __host__ __device__ inline float2 convert(const half_2 & u) {
        return __half22float2(u);
    }
};
template<> struct convertor<half_2, float2> {
    static __host__ __device__ inline half_2 convert(const float2 & u) {
        return __float22half2_rn(u);
    }
};
template<> struct convertor<bf16, half> {
    static __host__ __device__ inline bf16 convert(const half & u) {
        return __float2bfloat16(__half2float(u));
    }
};
template<> struct convertor<half, bf16> {
    static __host__ __device__ inline half convert(const bf16 & u) {
        return __float2half(__bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static __host__ __device__ inline bf16_2 convert(const half_2 & u) {
        return __float22bfloat162_rn(__half22float2(u));
    }
};
template<> struct convertor<half_2, bf16_2> {
    static __host__ __device__ inline half_2 convert(const bf16_2 & u) {
        return __float22half2_rn(__bfloat1622float2(u));
    }
};
template<> struct convertor<fp8e4m3_4, float4> {
    static __host__ __device__ inline fp8e4m3_4 convert(const float4& u) {
        return fp8e4m3_4(u);
    }
};
template<> struct convertor<float4, fp8e4m3_4> {
    static __host__ __device__ inline float4 convert(const fp8e4m3_4& u) {
        fp8e4m3 *vals = reinterpret_cast<fp8e4m3*>(const_cast<fp8e4m3_4*>(&u));
        return make_float4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]));
    }
};
template<> struct convertor<fp8e4m3_2, float2> {
    static __host__ __device__ inline fp8e4m3_2 convert(const float2& u) {
        return fp8e4m3_2(u);
    }
};
template<> struct convertor<float2, fp8e4m3_2> {
    static __host__ __device__ inline float2 convert(const fp8e4m3_2& u) {
        fp8e4m3 *vals = reinterpret_cast<fp8e4m3*>(const_cast<fp8e4m3_2*>(&u));
        return make_float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e4m3, float> {
    static __host__ __device__ inline fp8e4m3 convert(const float & u) {
        return fp8e4m3(u);
    }
};
template<> struct convertor<float, fp8e4m3> {
    static __host__ __device__ inline float convert(const fp8e4m3 & u) {
        return float(u);
    }
};

template<typename T, typename U> struct convertor<T, U> {
    static __host__ __device__ inline T convert(const U & u) {
        return (T)u;
    }
};

} // namespace base_types
} // namespace kittens
