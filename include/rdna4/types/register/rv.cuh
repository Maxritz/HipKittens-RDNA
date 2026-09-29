/**
 * @file
 * @brief Register vectors for computations on axes (RDNA4 gfx1201).
 *
 * Same structure as CDNA4's rv, adapted for wave-32.
 * The rv identifier lives in ducks::rv, same as CDNA5.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rv_layout.cuh"
#include "rt_shape.cuh"

namespace kittens {

namespace ducks {
namespace rv {
struct identifier {};
}
}

template<typename _T, size_t _length, size_t _tile_length, ducks::rt_shape::all _shape, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct rv {
    using identifier = ducks::rv::identifier;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using shape = _shape;
    using layout = _layout;
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;
    static constexpr bool is_ortho = std::is_same_v<layout, ducks::rv_layout::ortho>;
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = std::conditional_t<is_naive || is_ortho, T, T2>;
    static constexpr int packing = kittens::base_types::packing<dtype>::num();

    static constexpr int length = _length;
    static_assert(length % _tile_length == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = _length / _tile_length;
    static constexpr int inner_dim = is_naive ? ((length + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS) : (is_ortho ? 1 : _shape::elements_per_thread / packing);
    static constexpr int outer_dim = is_naive ? 1 : tiles;

    static constexpr int elements_per_thread = _shape::elements_per_thread;
    static constexpr int reductions = _tile_length;
    static constexpr int threads_per_reduction = reductions / elements_per_thread;
    static constexpr int aligned_threads = kittens::WARP_THREADS / threads_per_reduction;
    static constexpr int stride = _shape::stride;
    static constexpr int packed_per_stride = stride / packing;
    static constexpr int elements_per_stride_group = threads_per_reduction * stride;
    static constexpr int strides_per_tile = reductions / elements_per_stride_group;

    dtype data[outer_dim][inner_dim];

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; }
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; }
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; }
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; }
};

namespace ducks {
namespace rv {

template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template<typename T> concept tile_layout = align_layout<T> || ortho_layout<T>;

}
}

template<int _l, int _tile_length, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, _tile_length, shape, layout>;
template<int _l, int _tile_length, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_bf = rv<bf16,  _l, _tile_length, shape, layout>;
template<int _l, int _tile_length, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_hf = rv<half,  _l, _tile_length, shape, layout>;

template<typename _T, int _l> using rv_naive = rv<_T,  _l, _l, ducks::rt_shape::rt_16x16, ducks::rv_layout::naive>;

} // namespace kittens
