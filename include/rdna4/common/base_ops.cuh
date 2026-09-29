/**
 * @file
 * @brief Basic operations on generic types (RDNA4 gfx1201).
 *
 * Same as CDNA5 — these operate on base_types, no architecture-specific code.
 */

#pragma once

#include <limits>
#include "base_types.cuh"

namespace kittens {
namespace base_ops {

struct zero {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::zero(); }
};
struct ones {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::ones(); }
};
struct pos_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};
struct neg_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};
struct max {
    template<typename T, typename... args> __device__ static inline T op(const T &a, const T &b) { return ::max(a, b); }
};
struct min {
    template<typename T, typename... args> __device__ static inline T op(const T &a, const T &b) { return ::min(a, b); }
};
struct abs {
    template<typename T, typename... args> __device__ static inline T op(const T &a) { return ::abs(a); }
};

template<typename T>
__device__ static inline T exp2(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        float r;
        asm volatile("v_exp_f32_e32 %0, %1" : "=v"(r) : "v"(x));
        return r;
    } else {
        return ::exp2(x);
    }
}

template<typename T>
__device__ static inline T exp(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        return exp2<T>(M_LN2 * x);
    } else {
        return ::exp(x);
    }
}

template<typename T>
__device__ static inline T log2(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        float r;
        asm volatile("v_log_f32_e32 %0, %1" : "=v"(r) : "v"(x));
        return r;
    } else {
        return ::log2(x);
    }
}

template<typename T>
__device__ static inline T gelu(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        return (x * (1.0f + ::tanh(0.7978845608f * (x + 0.044719f * x * x * x)))) * 0.5f;
    } else {
        return (x * (1.0f + std::tanh(0.7978845608f * (x + 0.044719f * x * x * x)))) * 0.5f;
    }
}

template<typename T>
__device__ static inline T dgelu(const T& x) {
    if constexpr (std::is_same_v<T, float>) {
        float t = 0.7978845608f * (x + 0.044719f * x * x * x);
        float s = 1.0f + std::tanh(t);
        return 0.5f * (1.0f + std::tanh(t) + x * 0.5f * s * (1.0f - std::tanh(t)) * 0.7978845608f * (1.0f + 3.0f * 0.044719f * x * x));
    } else {
        float t = 0.7978845608f * (x + 0.044719f * x * x * x);
        float s = 1.0f + std::tanh(t);
        return 0.5f * (1.0f + std::tanh(t) + x * 0.5f * s * (1.0f - std::tanh(t)) * 0.7978845608f * (1.0f + 3.0f * 0.044719f * x * x));
    }
}

} // namespace base_ops

template<typename T> using op_zero = base_ops::zero;
template<typename T> using op_ones = base_ops::ones;
template<typename T> using op_pos_infty = base_ops::pos_infty;
template<typename T> using op_neg_infty = base_ops::neg_infty;

template<typename op, typename T, typename... args>
__device__ static inline T run_op(const T &param, args... as) {
    return op::template op<T, args...>(param, as...);
}

template<int NUM_THREADS=8>
struct reduce_add {
    template<typename T>
    __device__ static inline T run(const T &v, int thread) {
        T acc = v;
        #pragma unroll
        for (int i = NUM_THREADS / 2; i > 0; i /= 2) {
            acc += __shfl_down_sync(MASK_ALL, v, i, 64);
        }
        return acc;
    }
};

template<int NUM_THREADS=8>
struct reduce_add_2d {
    template<typename T>
    __device__ static inline T run(const T &v, int thread) {
        return reduce_add<NUM_THREADS>::run(v, thread);
    }
};

template<typename T> struct base_type {
    static constexpr bool has_fast_mul = false;
};
template<> struct base_type<float> {
    static constexpr bool has_fast_mul = true;
};
template<> struct base_type<half> {
    static constexpr bool has_fast_mul = true;
};

} // namespace kittens
