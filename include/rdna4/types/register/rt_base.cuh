/**
 * @file
 * @brief The basic register tile on which larger register tiles are built (RDNA4 gfx1201).
 *
 * Mirrors CDNA5's rt_base. The key difference is RDNA4 downstream:
 * - No a[] accumulator registers (everything in v[] VGPRs).
 * - WMMA 16x16x16 (K=16) intrinsics instead of MFMA.
 */

#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"
#include "rt_shape.cuh"
#include "rv_layout.cuh"

namespace kittens {

namespace ducks {
namespace rt_base {
struct identifier {};
}
}

/**
 * @brief Basic tile structure for computation in registers.
 *
 * @tparam _T The data type used for the matrix elements.
 * @tparam _layout The layout of the base tile, either row-major or column-major.
 * @tparam _shape The shape of the base tile (e.g. rt_16x16, rt_16x16_wmma_k16).
 */
template<typename _T, ducks::rt_layout::all _layout, ducks::rt_shape::all _shape> struct rt_base {
    using identifier = ducks::rt_base::identifier;
    using layout = _layout;
    using shape = _shape;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2;

    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2> || std::is_same_v<dtype, fp8e4m3_4>,
        "rt_base was provided an unsupported type."
    );

    static constexpr int rows = _shape::rows;
    static constexpr int cols = _shape::cols;
    static constexpr int stride = _shape::stride;
    static constexpr int num_elements = _shape::num_elements;
    static constexpr int elements_per_thread = _shape::elements_per_thread;
    static constexpr int num_strides = _shape::num_strides;

    static constexpr int reductions = std::is_same_v<layout, ducks::rt_layout::row> ? cols : rows;
    static constexpr int threads_per_reduction = reductions / elements_per_thread;
    static constexpr int elements_per_stride_group = threads_per_reduction * stride;

    static_assert(num_elements % stride == 0, "num_elements must be divisible by stride");

    static constexpr int num_packed = base_types::packing<dtype>::num();
    static constexpr int packed_per_thread    = (elements_per_thread / num_packed);
    static constexpr int packed_per_stride    = (stride / num_packed);
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4;

    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>;
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>;

    dtype data[packed_per_thread];
};

namespace ducks {
namespace rt_base {
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}

template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16> using rt_base_fl = rt_base<float, L, S>;
template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16> using rt_base_bf = rt_base<bf16,  L, S>;
template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16> using rt_base_hf = rt_base<half,  L, S>;

} // namespace kittens
