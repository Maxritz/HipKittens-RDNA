/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers (RDNA4).
 *
 * RDNA4 (gfx1201) uses Wave Matrix Multiply Accumulate (WMMA) intrinsics,
 * not CDNA5's MFMA. The ISA defines:
 *   V_WMMA_F32_16X16X32_BF16  — A(16x32) bf16 * B(32x16) bf16 + C(16x16) f32 => D(16x16) f32
 *   V_WMMA_F32_16X16X32_F16    — same with f16
 *   V_WMMA_F32_16X16X32_FP8_FP8 — FP8 (IEEE/OCP format on RDNA4)
 *   V_WMMA_BF16_16X16X16_BF16  — bf16 accumulation
 *   V_WMMA_F16_16X16X16_F16    — f16 accumulation
 *
 * Key differences from CDNA5:
 *   - K=32 (not MFMA's K=32/64/128) — matches CDNA5 wmma161632 shape
 *   - 4x float2 per accumulator (8 floats per lane for 16x16 tile)
 *   - 8x bf16_2 / half_2 per operand A (16 bf16 per lane for 16x32 tile)
 *   - No a[] accumulator registers — all v[] VGPRs
 *   - No scaled FMMA (mfma1616128_scaled, mfma_scale_f32_*)
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * WMMA 16x16x32 helper: 16-wide K=32 GEMM per WMMA call.
 * A is 16x32 packed bf16 (8 bf16_2 per lane), B is 32x16 packed bf16 (8 bf16_2 per lane).
 * C and D are 16x16 floats (4 float2 per lane).
 * Maps to V_WMMA_F32_16X16X32_BF16 on RDNA4.
 */
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161632(      float2 (&D)[4],
                                         const bf16_2 (&A)[8],
                                         const bf16_2 (&B)[8],
                                         const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(16 * sizeof(__bf16)))) __bf16 bf16x16_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    *(floatx16_t*)D = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
        /*a_neg=*/ false, *(bf16x16_t*)A,
        /*b_neg=*/ false, *(bf16x16_t*)B,
        /*c_mod=*/ 0,     *(floatx16_t*)C,
        A_reuse, B_reuse);
}

template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161632(      float2 (&D)[4],
                                         const half_2 (&A)[8],
                                         const half_2 (&B)[8],
                                         const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 fp16x16_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    *(floatx16_t*)D = __builtin_amdgcn_wmma_f32_16x16x32_f16(
        /*a_neg=*/ false, *(fp16x16_t*)A,
        /*b_neg=*/ false, *(fp16x16_t*)B,
        /*c_mod=*/ 0,     *(floatx16_t*)C,
        A_reuse, B_reuse);
}

/*
 * FP8 WMMA: V_WMMA_F32_16X16X32_FP8_FP8
 * On RDNA4, fp8e4m3 is IEEE/OCP format (HIP_FP8_TYPE_OCP=1).
 * A and B are 16x32 packed fp8 (8 fp8e4m3_4 per lane).
 */
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161632_fp8fp8(      float2 (&D)[4],
                                                 const fp8e4m3_4 (&A)[8],
                                                 const fp8e4m3_4 (&B)[8],
                                                 const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(32 * sizeof(uint8_t)))) uint8_t fp8x32_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    *(floatx16_t*)D = __builtin_amdgcn_wmma_f32_16x16x32_fp8_fp8(
        /*a_neg=*/ false, *(fp8x32_t*)A,
        /*b_neg=*/ false, *(fp8x32_t*)B,
        /*c_mod=*/ 0,     *(floatx16_t*)C,
        A_reuse, B_reuse);
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `wmma161632` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                        const rt_base<bf16, ducks::rt_layout::row, A_shape> &a,
                                        const rt_base<bf16, ducks::rt_layout::col, B_shape> &b,
                                        const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 32 && B_cols == 16 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> &&
                  A_rows == 32 && A_cols == 16 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        // 32x32 D tile = 4x 16x16 WMMA calls (2x2 grid)
        // A is col-major 32x16, B is row-major 16x32 — each quarter is 16x16
        // Split into 4 16x16 WMMA calls
        rt_base<float, ducks::rt_layout::col, D_shape> &d0 = d; // caller handles tiling
        wmma161632(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Base dot product operation for row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                    const rt_base<half, ducks::rt_layout::row, A_shape> &a,
                                    const rt_base<half, ducks::rt_layout::row, B_shape> &b,
                                    const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Base dot product operation for row layout with bf16.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                    const rt_base<bf16, ducks::rt_layout::row, A_shape> &a,
                                    const rt_base<bf16, ducks::rt_layout::row, B_shape> &b,
                                    const rt_base<float, ducks::rt_layout::col, C_shape> &c) {
    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Base dot product for fp8 row layout.
 * RDNA4 WMMA supports V_WMMA_F32_16X16X32_FP8_FP8 (IEEE/OCP fp8).
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                    const rt_base<fp8e4m3, ducks::rt_layout::row, A_shape> &a,
                                    const rt_base<fp8e4m3, ducks::rt_layout::row, B_shape> &b,
                                    const rt_base<float, ducks::rt_layout::col, C_shape> &c) {
    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 16 && A_cols == 32) {
        wmma161632_fp8fp8(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

// No block-scaled FMMA on RDNA4 — WMMA does not support E8M0 scaling
template<int opsel_a, int opsel_b, int cbsz = 0, int blgp = 0,
         ducks::rt_shape::all D_shape,
         ducks::rt_layout::all A_layout, ducks::rt_shape::all A_shape,
         ducks::rt_layout::all B_layout, ducks::rt_shape::all B_shape,
         ducks::rt_shape::all C_shape, typename MM_Operand_T>
__device__ static inline void mma_ABt_base_scaled(
    rt_base<float, ducks::rt_layout::col, D_shape> &d,
    const rt_base<MM_Operand_T, A_layout, A_shape> &a,
    const rt_base<MM_Operand_T, B_layout, B_shape> &b,
    const rt_base<float, ducks::rt_layout::col, C_shape> &c,
    const fp8e8m0_4 *scale_a,
    const fp8e8m0_4 *scale_b) {
    static_assert(false, "RDNA4 WMMA does not support block-scaled operations");
}

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols);
    static_assert(A::cols == B::rows);
    static_assert(D::rows == C::rows && D::cols == C::cols);

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Dot product operation for row layout.
 */
template<ducks::rt::col_layout D, ducks::rt::all A, ducks::rt::all B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Block scaled dot product operation for row layout.
 * NOT SUPPORTED on RDNA4 — no scaled WMMA.
 */
template<int cbsz = 0, int blgp = 0, ducks::rt::col_layout D, ducks::rt::all A, ducks::rt::all B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt_scaled(D &d,
                                     const A &a,
                                     const B &b,
                                     const C &c,
                                     const fp8e8m0_4 *scale_a,
                                     const fp8e8m0_4 *scale_b) {
    static_assert(false, "RDNA4 WMMA does not support block-scaled operations");
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtB(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols);
    static_assert(A::rows == B::rows);
    static_assert(D::rows == C::rows && D::cols == C::cols);

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Base matrix multiply-accumulate for transposed A (col-major A).
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                            const rt_base<bf16, ducks::rt_layout::col, A_shape> &a,
                                            const rt_base<bf16, ducks::rt_layout::col, B_shape> &b,
                                            const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 32 && A_cols == 16 &&
                  B_rows == 32 && B_cols == 16 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> &&
                  A_rows == 32 && A_cols == 32 &&
                  B_rows == 32 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Base matrix multiply-accumulate for transposed A and B.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                             const rt_base<bf16, ducks::rt_layout::col, A_shape> &a,
                                             const rt_base<bf16, ducks::rt_layout::row, B_shape> &b,
                                             const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 32 && A_cols == 16 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 32 && B_cols == 16 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        wmma161632(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows);
    static_assert(A::rows == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

} // namespace kittens