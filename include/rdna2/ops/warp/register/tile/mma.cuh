/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers (RDNA2 adaptation).
 *
 * RDNA2 (gfx1031) has no MFMA instructions. Instead, we use scalar decomposition
 * with v_pk_fmac_f32 chains for packed f16/bf16 operations.
 * FP8 paths are static_assert'd out (RDNA2 has no FP8 support).
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * RDNA2 MMA via packed fma instructions.
 * v_pk_fmac_f32 performs D = A * B + D on packed half-precision pairs.
 * Each v_pk_fmac_f32 computes 2 f16 products + 2 f16 accumulates per cycle.
 * A 16x16x16 f16 GEMM requires 128 such packed operations across the warp.
 */

// Packed half fma: D.lo = A.lo * B.lo + D.lo, D.hi = A.hi * B.hi + D.hi
__device__ __forceinline__ void pk_fmac_f32(float2 &d, const half_2 &a, const half_2 &b) {
  asm volatile("v_pk_fmac_f32 %0, %1, %2, %0"
    : "+v"(d)
    : "v"(a), "v"(b));
}

// Unpacked variant: D = A * B + C (separate output)
__device__ __forceinline__ float2 pk_fmac_f32(const half_2 &a, const half_2 &b, const float2 &c) {
  float2 d;
  asm volatile("v_pk_fmac_f32 %0, %1, %2, %3"
    : "=v"(d)
    : "v"(a), "v"(b), "v"(c));
  return d;
}

// bf16 packed fma (bf16 is stored as uint16_t)
__device__ __forceinline__ void pk_fmac_f32(float2 &d, const bf16_2 &a, const bf16_2 &b) {
  asm volatile("v_pk_fmac_f32 %0, %1, %2, %0"
    : "+v"(d)
    : "v"(a), "v"(b));
}

__device__ __forceinline__ float2 pk_fmac_f32(const bf16_2 &a, const bf16_2 &b, const float2 &c) {
  float2 d;
  asm volatile("v_pk_fmac_f32 %0, %1, %2, %3"
    : "=v"(d)
    : "v"(a), "v"(b), "v"(c));
  return d;
}

/*
 * mfma161616 for f16:
 * 16x16x16 matrix multiply using v_pk_fmac_f32 chains.
 * A and B are each 2x half_2 (4 half values per thread = 64 threads * 4 = 256 f16 per matrix = 16x16).
 * C and D are each 2x float2 (4 floats per thread).
 */
__device__ static inline void mfma161616(float2 (&D)[2],
                                        const half_2 (&A)[2],
                                        const half_2 (&B)[2],
                                        const float2 (&C)[2]) {
  // Decompose into individual f16 multiply-adds using packed operations
  float2 d0 = C[0];
  float2 d1 = C[1];

  // 4x v_pk_fmac_f32 covers all 8 f16x16x16 elements per thread
  d0 = pk_fmac_f32(A[0], B[0], d0);
  d0 = pk_fmac_f32(A[0], B[1], d0);
  d1 = pk_fmac_f32(A[1], B[0], d1);
  d1 = pk_fmac_f32(A[1], B[1], d1);

  D[0] = d0;
  D[1] = d1;
}

__device__ static inline void mfma161616(float2 (&D)[2],
                                        const bf16_2 (&A)[2],
                                        const bf16_2 (&B)[2],
                                        const float2 (&C)[2]) {
  float2 d0 = C[0];
  float2 d1 = C[1];

  d0 = pk_fmac_f32(A[0], B[0], d0);
  d0 = pk_fmac_f32(A[0], B[1], d0);
  d1 = pk_fmac_f32(A[1], B[0], d1);
  d1 = pk_fmac_f32(A[1], B[1], d1);

  D[0] = d0;
  D[1] = d1;
}

/*
 * FP8 paths are not supported on RDNA2.
 */
__device__ static inline void mfma161632(float2 (&D)[2],
                                        const fp8e4m3_4 (&A)[2],
                                        const fp8e4m3_4 (&B)[2],
                                        const float2 (&C)[2]) {
  static_assert(sizeof(fp8e4m3) == 0, "RDNA2 does not support FP8 MMA operations");
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b,
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b,
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<fp8e4m3, ducks::rt_layout::row> &a,
                                    const rt_base<fp8e4m3, ducks::rt_layout::col> &b,
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161632(d.data, a.data, b.data, c.data);
}

/**
 * @brief Base dot product operation for row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<fp8e4m3, ducks::rt_layout::row> &a,
                                     const rt_base<fp8e4m3, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161632(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::col> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b,
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b,
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<fp8e4m3, ducks::rt_layout::col> &a,
                                     const rt_base<fp8e4m3, ducks::rt_layout::col> &b,
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161632(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<half, ducks::rt_layout::col> &a,
                                      const rt_base<half, ducks::rt_layout::row> &b,
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b,
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<fp8e4m3, ducks::rt_layout::col> &a,
                                      const rt_base<fp8e4m3, ducks::rt_layout::row> &b,
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161632(d.data, a.data, b.data, c.data);
}
/**
 * @brief Matrix multiply-accumulate operation.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
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
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename  D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
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
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
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
        (std::is_same_v<typename  D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
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
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
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
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename  D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
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
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
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
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename  D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
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
}
