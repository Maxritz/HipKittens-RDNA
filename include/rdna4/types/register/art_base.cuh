/**
 * @file
 * @brief The basic 16x16 register tile with assembly mode (RDNA4 gfx1201).
 *
 * Mirrors CDNA5's art_base, but on RDNA4 all registers are VGPR (v[]),
 * there are no accumulator registers (a[]). The shape system uses
 * WMMA 16x16x16 (K=16).
 */

#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"
#include "rt_shape.cuh"
#include "rv_layout.cuh"

namespace kittens {

namespace ducks {
namespace art_base {
struct identifier {};
}
}

template<typename _T, ducks::rt_layout::all _layout, ducks::rt_shape::all _shape, typename _register_range>
struct art_base {
    using identifier = ducks::art_base::identifier;
    using layout = _layout;
    using shape = _shape;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2;
    using register_range = _register_range;

    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2> || std::is_same_v<dtype, fp8e4m3_4>,
        "art_base was provided an unsupported type."
    );

    static constexpr int rows                 = shape::rows;
    static constexpr int cols                 = shape::cols;
    static constexpr int stride               = shape::stride;
    static constexpr int num_elements         = rows*cols;
    static constexpr int elements_per_thread  = num_elements / kittens::WARP_THREADS;
    static constexpr int num_strides          = shape::num_strides;

    static constexpr int reductions = std::is_same_v<layout, ducks::rt_layout::row> ? cols : rows;
    static constexpr int threads_per_reduction = reductions / elements_per_thread;
    static constexpr int elements_per_stride_group = threads_per_reduction * stride;

    static_assert(num_elements % stride == 0, "num_elements must be divisible by stride");

    static constexpr int packed_per_thread    = (elements_per_thread / base_types::packing<dtype>::num());
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4;
    static constexpr int registers_per_stride = registers_per_thread / num_strides;

    static_assert(register_range::size == registers_per_thread,
                  "Register range size must match registers_per_thread for art_base");

    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>;
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>;

    register_range registers;
};

namespace ducks {
namespace art_base {
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}

namespace ducks { namespace art { template<int, int> struct range; } }

template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16_wmma_k16, typename R=ducks::art::range<0, 1>> using art_base_fl = art_base<float, L, S, R>;
template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16_wmma_k16, typename R=ducks::art::range<0, 1>> using art_base_bf = art_base<bf16,  L, S, R>;
template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16_wmma_k16, typename R=ducks::art::range<0, 1>> using art_base_hf = art_base<half,  L, S, R>;

} // namespace kittens
