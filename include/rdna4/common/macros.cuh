/**
 * @file
 * @brief Macro-style intrinsics for RDNA4 (gfx1201).
 *
 * RDNA4 has no accumulator registers (a[]); all VGPR operations use v[].
 * RDNA4 uses WMMA 16x16x16 (K=16) intrinsics, not CDNA's MFMA.
 */

#pragma once

#include "base_types.cuh"
#include "util.cuh"

namespace kittens {
namespace macros {

// VGPR-only clobber for RDNA4 (no a[] registers)
#define CLOBBER_VREG_CASE(N) case N: asm volatile("" ::: "v" #N); break;

template<int GPR>
__device__ __forceinline__ void clobber_gpr() {
    constexpr int reg = GPR;
    switch (reg) {
        CLOBBER_VREG_CASE(0)   CLOBBER_VREG_CASE(1)   CLOBBER_VREG_CASE(2)   CLOBBER_VREG_CASE(3)
        CLOBBER_VREG_CASE(4)   CLOBBER_VREG_CASE(5)   CLOBBER_VREG_CASE(6)   CLOBBER_VREG_CASE(7)
        CLOBBER_VREG_CASE(8)   CLOBBER_VREG_CASE(9)   CLOBBER_VREG_CASE(10)  CLOBBER_VREG_CASE(11)
        CLOBBER_VREG_CASE(12)  CLOBBER_VREG_CASE(13)  CLOBBER_VREG_CASE(14)  CLOBBER_VREG_CASE(15)
        CLOBBER_VREG_CASE(16)  CLOBBER_VREG_CASE(17)  CLOBBER_VREG_CASE(18)  CLOBBER_VREG_CASE(19)
        CLOBBER_VREG_CASE(20)  CLOBBER_VREG_CASE(21)  CLOBBER_VREG_CASE(22)  CLOBBER_VREG_CASE(23)
        CLOBBER_VREG_CASE(24)  CLOBBER_VREG_CASE(25)  CLOBBER_VREG_CASE(26)  CLOBBER_VREG_CASE(27)
        CLOBBER_VREG_CASE(28)  CLOBBER_VREG_CASE(29)  CLOBBER_VREG_CASE(30)  CLOBBER_VREG_CASE(31)
        CLOBBER_VREG_CASE(32)  CLOBBER_VREG_CASE(33)  CLOBBER_VREG_CASE(34)  CLOBBER_VREG_CASE(35)
        CLOBBER_VREG_CASE(36)  CLOBBER_VREG_CASE(37)  CLOBBER_VREG_CASE(38)  CLOBBER_VREG_CASE(39)
        CLOBBER_VREG_CASE(40)  CLOBBER_VREG_CASE(41)  CLOBBER_VREG_CASE(42)  CLOBBER_VREG_CASE(43)
        CLOBBER_VREG_CASE(44)  CLOBBER_VREG_CASE(45)  CLOBBER_VREG_CASE(46)  CLOBBER_VREG_CASE(47)
        CLOBBER_VREG_CASE(48)  CLOBBER_VREG_CASE(49)  CLOBBER_VREG_CASE(50)  CLOBBER_VREG_CASE(51)
        CLOBBER_VREG_CASE(52)  CLOBBER_VREG_CASE(53)  CLOBBER_VREG_CASE(54)  CLOBBER_VREG_CASE(55)
        CLOBBER_VREG_CASE(56)  CLOBBER_VREG_CASE(57)  CLOBBER_VREG_CASE(58)  CLOBBER_VREG_CASE(59)
        CLOBBER_VREG_CASE(60)  CLOBBER_VREG_CASE(61)  CLOBBER_VREG_CASE(62)  CLOBBER_VREG_CASE(63)
        CLOBBER_VREG_CASE(64)  CLOBBER_VREG_CASE(65)  CLOBBER_VREG_CASE(66)  CLOBBER_VREG_CASE(67)
        CLOBBER_VREG_CASE(68)  CLOBBER_VREG_CASE(69)  CLOBBER_VREG_CASE(70)  CLOBBER_VREG_CASE(71)
        CLOBBER_VREG_CASE(72)  CLOBBER_VREG_CASE(73)  CLOBBER_VREG_CASE(74)  CLOBBER_VREG_CASE(75)
        CLOBBER_VREG_CASE(76)  CLOBBER_VREG_CASE(77)  CLOBBER_VREG_CASE(78)  CLOBBER_VREG_CASE(79)
        CLOBBER_VREG_CASE(80)  CLOBBER_VREG_CASE(81)  CLOBBER_VREG_CASE(82)  CLOBBER_VREG_CASE(83)
        CLOBBER_VREG_CASE(84)  CLOBBER_VREG_CASE(85)  CLOBBER_VREG_CASE(86)  CLOBBER_VREG_CASE(87)
        CLOBBER_VREG_CASE(88)  CLOBBER_VREG_CASE(89)  CLOBBER_VREG_CASE(90)  CLOBBER_VREG_CASE(91)
        CLOBBER_VREG_CASE(92)  CLOBBER_VREG_CASE(93)  CLOBBER_VREG_CASE(94)  CLOBBER_VREG_CASE(95)
        CLOBBER_VREG_CASE(96)  CLOBBER_VREG_CASE(97)  CLOBBER_VREG_CASE(98)  CLOBBER_VREG_CASE(99)
        CLOBBER_VREG_CASE(100) CLOBBER_VREG_CASE(101) CLOBBER_VREG_CASE(102) CLOBBER_VREG_CASE(103)
        CLOBBER_VREG_CASE(104) CLOBBER_VREG_CASE(105) CLOBBER_VREG_CASE(106) CLOBBER_VREG_CASE(107)
        CLOBBER_VREG_CASE(108) CLOBBER_VREG_CASE(109) CLOBBER_VREG_CASE(110) CLOBBER_VREG_CASE(111)
        CLOBBER_VREG_CASE(112) CLOBBER_VREG_CASE(113) CLOBBER_VREG_CASE(114) CLOBBER_VREG_CASE(115)
        CLOBBER_VREG_CASE(116) CLOBBER_VREG_CASE(117) CLOBBER_VREG_CASE(118) CLOBBER_VREG_CASE(119)
        CLOBBER_VREG_CASE(120) CLOBBER_VREG_CASE(121) CLOBBER_VREG_CASE(122) CLOBBER_VREG_CASE(123)
        CLOBBER_VREG_CASE(124) CLOBBER_VREG_CASE(125) CLOBBER_VREG_CASE(126) CLOBBER_VREG_CASE(127)
        CLOBBER_VREG_CASE(128) CLOBBER_VREG_CASE(129) CLOBBER_VREG_CASE(130) CLOBBER_VREG_CASE(131)
        CLOBBER_VREG_CASE(132) CLOBBER_VREG_CASE(133) CLOBBER_VREG_CASE(134) CLOBBER_VREG_CASE(135)
        CLOBBER_VREG_CASE(136) CLOBBER_VREG_CASE(137) CLOBBER_VREG_CASE(138) CLOBBER_VREG_CASE(139)
        CLOBBER_VREG_CASE(140) CLOBBER_VREG_CASE(141) CLOBBER_VREG_CASE(142) CLOBBER_VREG_CASE(143)
        CLOBBER_VREG_CASE(144) CLOBBER_VREG_CASE(145) CLOBBER_VREG_CASE(146) CLOBBER_VREG_CASE(147)
        CLOBBER_VREG_CASE(148) CLOBBER_VREG_CASE(149) CLOBBER_VREG_CASE(150) CLOBBER_VREG_CASE(151)
        CLOBBER_VREG_CASE(152) CLOBBER_VREG_CASE(153) CLOBBER_VREG_CASE(154) CLOBBER_VREG_CASE(155)
        CLOBBER_VREG_CASE(156) CLOBBER_VREG_CASE(157) CLOBBER_VREG_CASE(158) CLOBBER_VREG_CASE(159)
        CLOBBER_VREG_CASE(160) CLOBBER_VREG_CASE(161) CLOBBER_VREG_CASE(162) CLOBBER_VREG_CASE(163)
        CLOBBER_VREG_CASE(164) CLOBBER_VREG_CASE(165) CLOBBER_VREG_CASE(166) CLOBBER_VREG_CASE(167)
        CLOBBER_VREG_CASE(168) CLOBBER_VREG_CASE(169) CLOBBER_VREG_CASE(170) CLOBBER_VREG_CASE(171)
        CLOBBER_VREG_CASE(172) CLOBBER_VREG_CASE(173) CLOBBER_VREG_CASE(174) CLOBBER_VREG_CASE(175)
        CLOBBER_VREG_CASE(176) CLOBBER_VREG_CASE(177) CLOBBER_VREG_CASE(178) CLOBBER_VREG_CASE(179)
        CLOBBER_VREG_CASE(180) CLOBBER_VREG_CASE(181) CLOBBER_VREG_CASE(182) CLOBBER_VREG_CASE(183)
        CLOBBER_VREG_CASE(184) CLOBBER_VREG_CASE(185) CLOBBER_VREG_CASE(186) CLOBBER_VREG_CASE(187)
        CLOBBER_VREG_CASE(188) CLOBBER_VREG_CASE(189) CLOBBER_VREG_CASE(190) CLOBBER_VREG_CASE(191)
        CLOBBER_VREG_CASE(192) CLOBBER_VREG_CASE(193) CLOBBER_VREG_CASE(194) CLOBBER_VREG_CASE(195)
        CLOBBER_VREG_CASE(196) CLOBBER_VREG_CASE(197) CLOBBER_VREG_CASE(198) CLOBBER_VREG_CASE(199)
        CLOBBER_VREG_CASE(200) CLOBBER_VREG_CASE(201) CLOBBER_VREG_CASE(202) CLOBBER_VREG_CASE(203)
        CLOBBER_VREG_CASE(204) CLOBBER_VREG_CASE(205) CLOBBER_VREG_CASE(206) CLOBBER_VREG_CASE(207)
        CLOBBER_VREG_CASE(208) CLOBBER_VREG_CASE(209) CLOBBER_VREG_CASE(210) CLOBBER_VREG_CASE(211)
        CLOBBER_VREG_CASE(212) CLOBBER_VREG_CASE(213) CLOBBER_VREG_CASE(214) CLOBBER_VREG_CASE(215)
        CLOBBER_VREG_CASE(216) CLOBBER_VREG_CASE(217) CLOBBER_VREG_CASE(218) CLOBBER_VREG_CASE(219)
        CLOBBER_VREG_CASE(220) CLOBBER_VREG_CASE(221) CLOBBER_VREG_CASE(222) CLOBBER_VREG_CASE(223)
        CLOBBER_VREG_CASE(224) CLOBBER_VREG_CASE(225) CLOBBER_VREG_CASE(226) CLOBBER_VREG_CASE(227)
        CLOBBER_VREG_CASE(228) CLOBBER_VREG_CASE(229) CLOBBER_VREG_CASE(230) CLOBBER_VREG_CASE(231)
        CLOBBER_VREG_CASE(232) CLOBBER_VREG_CASE(233) CLOBBER_VREG_CASE(234) CLOBBER_VREG_CASE(235)
        CLOBBER_VREG_CASE(236) CLOBBER_VREG_CASE(237) CLOBBER_VREG_CASE(238) CLOBBER_VREG_CASE(239)
        CLOBBER_VREG_CASE(240) CLOBBER_VREG_CASE(241) CLOBBER_VREG_CASE(242) CLOBBER_VREG_CASE(243)
        CLOBBER_VREG_CASE(244) CLOBBER_VREG_CASE(245) CLOBBER_VREG_CASE(246) CLOBBER_VREG_CASE(247)
        CLOBBER_VREG_CASE(248) CLOBBER_VREG_CASE(249) CLOBBER_VREG_CASE(250) CLOBBER_VREG_CASE(251)
        CLOBBER_VREG_CASE(252) CLOBBER_VREG_CASE(253) CLOBBER_VREG_CASE(254) CLOBBER_VREG_CASE(255)
    }
}
#undef CLOBBER_VREG_CASE

__device__ __forceinline__ constexpr uint32_t max_ds_inst_offset() { return (1u << 16) - 1; }
__device__ __forceinline__ constexpr uint32_t max_ds_pk2_inst_offset() { return (1u << 8) - 1; }
__device__ __forceinline__ constexpr uint32_t max_mubuf_inst_offset() { return (1u << 12) - 1; }

template<int GPR_START>
__device__ __forceinline__ void ds_read_b32(const uint32_t smem_ptr, const int i_offset) {
    asm volatile("ds_read_b32 v[%0], %1 offset:%2"
        :
        : "n"(GPR_START), "v"(smem_ptr), "i"(i_offset)
        : "memory");
}

template <typename T>
__device__ __forceinline__ void ds_read_b32(T& dst, const uint32_t smem_ptr, const int i_offset) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    asm volatile("ds_read_b32 %0, %1 offset:%2"
        : "=v"(dst)
        : "v"(smem_ptr), "i"(i_offset)
        : "memory");
}

template <typename T = u32x2>
__device__ __forceinline__ T ds_read_b64(const uint32_t smem_ptr, const int i_offset) {
    static_assert(sizeof(T) == sizeof(uint32_t) * 2);
    T result;
    asm volatile("ds_read_b64 %0, %1 offset:%2"
        : "=v"(result)
        : "v"(smem_ptr), "i"(i_offset)
        : "memory");
    return result;
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b64(const uint32_t smem_ptr, const int i_offset) {
    constexpr int GPR_END = GPR_START + 1;
    asm volatile("ds_read_b64 v[%0:%1], %2 offset:%3"
        :
        : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(i_offset)
        : "memory");
}

template <typename T = u32x4>
__device__ __forceinline__ T ds_read_b128(const uint32_t smem_ptr, const int i_offset) {
    static_assert(sizeof(T) == sizeof(uint32_t) * 4);
    T result;
    asm volatile("ds_read_b128 %0, %1 offset:%2"
        : "=v"(result)
        : "v"(smem_ptr), "i"(i_offset)
        : "memory");
    return result;
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b128(const uint32_t smem_ptr, const int i_offset) {
    constexpr int GPR_END = GPR_START + 3;
    asm volatile("ds_read_b128 v[%0:%1], %2 offset:%3"
        :
        : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(i_offset)
        : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void ds_write_b32(const uint32_t smem_ptr, const int i_offset) {
    asm volatile("ds_write_b32 %0, v[%1] offset:%2"
        :
        : "v"(smem_ptr), "n"(GPR_START), "i"(i_offset)
        : "memory");
}

template <typename T>
__device__ __forceinline__ void ds_write_b32(const T& val, const uint32_t smem_ptr, const int i_offset = 0) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    asm volatile("ds_write_b32 %0, %1 offset:%2"
        :
        : "v"(smem_ptr), "v"(val), "i"(i_offset)
        : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void ds_write_b64(const uint32_t smem_ptr, const int i_offset) {
    asm volatile("ds_write_b64 %0, v[%1:%2], offset:%3"
        :
        : "v"(smem_ptr), "n"(GPR_START), "n"(GPR_START + 1), "i"(i_offset)
        : "memory");
}

template <typename T>
__device__ __forceinline__ void ds_write_b64(const T& val, const uint32_t smem_ptr, const int i_offset = 0) {
    static_assert(sizeof(T) == 2 * sizeof(uint32_t));
    asm volatile("ds_write_b64 %0, %1 offset:%2"
        :
        : "v"(smem_ptr), "v"(val), "i"(i_offset)
        : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void ds_write_b128(const uint32_t smem_ptr, const int i_offset = 0) {
    asm volatile("ds_write_b128 %0, v[%1:%2], offset:%3"
        :
        : "v"(smem_ptr), "n"(GPR_START), "n"(GPR_START + 3), "i"(i_offset)
        : "memory");
}

template<typename T>
__device__ __forceinline__ void ds_write_b128(const T& value, const uint32_t smem_ptr, const int i_offset = 0) {
    static_assert(sizeof(T) == sizeof(u32x4));
    asm volatile("ds_write_b128 %0, %1 offset:%2"
        :
        : "v"(smem_ptr), "v"(value), "i"(i_offset)
        : "memory");
}

// WMMA 16x16x16 intrinsics (RDNA4 uses these, not MFMA)
template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void wmma_f32_16x16x16_f32(float4& D, const float4& A, const float4& B, const float4& C) {
    asm volatile("v_wmma_f32_16x16x16_f32 %0, %1, %2, %3"
        : "=v"(D)
        : "v"(A), "v"(B), "v"(C));
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void wmma_f32_16x16x16_bf16(float4& D, const float4& A, const float4& B, const float4& C) {
    asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %3"
        : "=v"(D)
        : "v"(A), "v"(B), "v"(C));
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void wmma_f32_16x16x16_fp8e4m3(float4& D, const float4& A, const float4& B, const float4& C) {
    asm volatile("v_wmma_f32_16x16x16_fp8 %0, %1, %2, %3"
        : "=v"(D)
        : "v"(A), "v"(B), "v"(C));
}

// Packed math for RDNA4
template<int GPR_START_A, int GPR_START_B>
__device__ __forceinline__ void pk_mul_f32(const float& param) {
    const float2 param2 = {param, param};
    asm volatile("v_pk_mul_f32 v[%0:%1], %4, v[%2:%3]"
        :
        : "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1), "v"(param2));
}

template<int GPR_START_A, int GPR_START_B>
__device__ __forceinline__ void pk_fma_f32(const float& param) {
    const float2 param2 = {param, param};
    asm volatile("v_pk_fma_f32 v[%0:%1], %4, v[%2:%3], v[%2:%3]"
        :
        : "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1), "v"(param2));
}

template<int GPR0, int GPR1>
__device__ __forceinline__ void exp2() {
    asm volatile("v_exp_f32_e32 v[%0], v[%1]"
        :
        : "n"(GPR0), "n"(GPR1));
}

template<int GPR0>
__device__ __forceinline__ void zero() {
    asm volatile("v_mov_b32 v[%0], 0"
        :
        : "n"(GPR0));
}

} // namespace macros
} // namespace kittens
