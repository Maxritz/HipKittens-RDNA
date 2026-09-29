/**
 * @file
 * @brief The primitives for register tiles with assembly mode (RDNA4 gfx1201).
 *
 * Mirrors CDNA5's art with register range management. On RDNA4, all
 * registers are VGPR (v[]) — there are no accumulator registers.
 */

#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "art_base.cuh"
#include "rv.cuh"

namespace kittens {

namespace ducks {
namespace art {

template <typename... Ts> struct type_list {
    static constexpr int size = sizeof...(Ts);
};

template <typename L1, typename L2> struct concat;
template <typename... A, typename... B>
struct concat<type_list<A...>, type_list<B...>> { using type = type_list<A..., B...>; };

template <typename TList> struct type_list_size;
template <typename... Ts>
struct type_list_size<type_list<Ts...>> {
    static constexpr int value = sizeof...(Ts);
};
template <typename TList>
static constexpr int type_list_size_v = type_list_size<TList>::value;

template <int L, int R>
struct range {
    static_assert(L <= R, "range requires L <= R");
    static constexpr int lo = L, hi = R;
    static constexpr int size = R - L + 1;
};

template <int L, int R, int N, bool Done = (L > R)>
struct split_one;

template <int L, int R, int N>
struct split_one<L, R, N, true> { using type = type_list<>; };

template <int L, int R, int N>
struct split_one<L, R, N, false> {
    static_assert(N > 0, "N must be > 0");
    static_assert(L + N - 1 <= R, "L + N - 1 must be <= R");
    static constexpr int end = L + N - 1;
    using head = range<L, end>;
    using tail = typename split_one<end + 1, R, N>::type;
    using type = typename concat<type_list<head>, tail>::type;
};

template <typename RList, int N> struct split_many;
template <int N>
struct split_many<type_list<>, N> { using type = type_list<>; };

template <typename R1, typename... Rs, int N>
struct split_many<type_list<R1, Rs...>, N> {
    using first = typename split_one<R1::lo, R1::hi, N>::type;
    using rest  = typename split_many<type_list<Rs...>, N>::type;
    using type  = typename concat<first, rest>::type;
};

template <typename RList, int N>
using split_many_t = typename split_many<RList, N>::type;

template <typename RangeList, int N>
struct get_nth_range;

template <typename R1, typename... Rs, int N>
struct get_nth_range<type_list<R1, Rs...>, N> {
    using type = typename std::conditional_t<N == 0, R1, typename get_nth_range<type_list<Rs...>, N-1>::type>;
};

template <typename R1, typename... Rs>
struct get_nth_range<type_list<R1, Rs...>, 0> {
    using type = R1;
};

template <typename RangeList, int N>
using get_nth_range_t = typename get_nth_range<RangeList, N>::type;

template <typename TList, int H, int W, int... Indices>
struct transpose_2d_impl;

template <typename TList, int H, int W>
struct transpose_2d_impl<TList, H, W> {
    using type = type_list<>;
};

template <typename TList, int H, int W, int I, int... Rest>
struct transpose_2d_impl<TList, H, W, I, Rest...> {
    static constexpr int r = I % H;
    static constexpr int c = I / H;
    static constexpr int src_idx = r * W + c;
    using current = type_list<get_nth_range_t<TList, src_idx>>;
    using rest = typename transpose_2d_impl<TList, H, W, Rest...>::type;
    using type = typename concat<current, rest>::type;
};

template <typename TList, int H, int W>
struct transpose_2d_helper {
    static_assert(type_list_size_v<TList> == H * W, "List size must equal H * W");
    template <int... Is>
    static auto make_impl(std::integer_sequence<int, Is...>)
        -> typename transpose_2d_impl<TList, H, W, Is...>::type;
    using type = decltype(make_impl(std::make_integer_sequence<int, H * W>{}));
};

template <typename TList, int H, int W>
using transpose_2d = typename transpose_2d_helper<TList, H, W>::type;

template<typename T>
concept register_range_t = requires {
    T::lo;
    T::hi;
    T::size;
};

template<typename RList>
__device__ inline static void clobber() {
    using registers = ducks::art::split_many_t<RList, 1>;
    [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
        ([&]<std::size_t R>() {
            macros::clobber_gpr<ducks::art::get_nth_range_t<registers, R>::lo>();
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<registers::size>{});
}

struct asm_identifier {};
} // namespace art
} // namespace ducks

template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row,
         ducks::rt_shape::all _shape=ducks::rt_shape::rt_16x16_wmma_k16,
         typename _register_ranges=ducks::art::type_list<ducks::art::range<0, 1>>>
struct art {
    using identifier = ducks::art::asm_identifier;
    using layout = _layout;
    using shape = _shape;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2;
    using register_ranges = _register_ranges;

    static constexpr int rows = _rows;
    static_assert(rows % art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::rows == 0, "Rows must be divisible by the tile size");
    static constexpr int cols = _cols;
    static_assert(cols % art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::cols == 0, "Columns must be divisible by the tile size");
    static constexpr int height = rows / art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::rows;
    static constexpr int width  = cols / art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::cols;

    static constexpr int base_tile_rows        = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::rows;
    static constexpr int base_tile_cols        = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::cols;
    static constexpr int base_tile_stride      = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::stride;
    static constexpr int base_tile_num_strides = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::num_strides;
    static constexpr int base_tile_reductions  = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::reductions;
    static constexpr int base_tile_threads_per_reduction = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::threads_per_reduction;
    static constexpr int base_tile_elements_per_stride_group = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::elements_per_stride_group;

    static constexpr int num_elements        = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::num_elements        * width * height;
    static constexpr int elements_per_thread = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::elements_per_thread * width * height;
    static constexpr int packed_per_thread   = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::packed_per_thread   * width * height;
    static constexpr int packed_per_base_tile    = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::packed_per_thread;
    static constexpr int elements_per_base_tile  = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::elements_per_thread;
    static constexpr int registers_per_stride = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::registers_per_stride;

    static_assert(ducks::art::type_list_size_v<register_ranges> == height * width,
        "Not enough register ranges provided for all base tiles in art");

    template<int Row, int Col>
    using base_tile_type = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, Row * width + Col>>;

    using row_vec = rv<T, cols, base_tile_cols, shape, typename art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::row_vec_layout>;
    using col_vec = rv<T, rows, base_tile_rows, shape, typename art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::col_vec_layout>;
};

namespace ducks {
namespace art {
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, asm_identifier>;

template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
}
}

template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row,
         ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16_wmma_k16,
         typename ranges=ducks::art::type_list<ducks::art::range<0, 1>>> using art_fl = art<float, _r, _c, layout, shape, ranges>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row,
         ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16_wmma_k16,
         typename ranges=ducks::art::type_list<ducks::art::range<0, 1>>> using art_bf = art<bf16,  _r, _c, layout, shape, ranges>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row,
         ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16_wmma_k16,
         typename ranges=ducks::art::type_list<ducks::art::range<0, 1>>> using art_hf = art<half,  _r, _c, layout, shape, ranges>;

} // namespace kittens
