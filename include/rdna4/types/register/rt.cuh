/**
 * @file
 * @brief The main register tile struct for RDNA4 (gfx1201).
 *
 * Mirrors CDNA5's rt: composed of tiles[height][width] of rt_base subtiles.
 * For WMMA 16x16x16 (K=16), the default shape is rt_16x16_wmma_k16.
 * No a[] registers on RDNA4 — everything is in v[] VGPRs.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

#include "rt_layout.cuh"
#include "rt_base.cuh"
#include "rt_shape.cuh"
#include "rv.cuh"

namespace kittens {

namespace ducks {
namespace rt {
struct identifier {};
}
}

/**
 * @brief Main tile structure for manipulating data in registers.
 *
 * @tparam _T The data type used for the matrix elements.
 * @tparam _rows The total number of rows.
 * @tparam _cols The total number of columns.
 * @tparam _layout The layout of the internal base tiles (row-major or column-major).
 * @tparam _shape The shape of the base subtile.
 */
template<typename _T, int _rows, int _cols,
         ducks::rt_layout::all _layout = ducks::rt_layout::row,
         ducks::rt_shape::all _shape = ducks::rt_shape::rt_16x16_wmma_k16>
struct rt {
    using identifier = ducks::rt::identifier;
    using layout = _layout;
    using shape = _shape;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2;

    static constexpr int rows                = _rows;
    static_assert(rows % rt_base<T, layout, shape>::rows == 0, "Rows must be divisible by the tile size");
    static constexpr int cols                = _cols;
    static_assert(cols % rt_base<T, layout, shape>::cols == 0, "Columns must be divisible by the tile size");
    static constexpr int height              = rows / rt_base<T, layout, shape>::rows;
    static constexpr int width               = cols / rt_base<T, layout, shape>::cols;

    static constexpr int base_tile_rows        = rt_base<T, layout, shape>::rows;
    static constexpr int base_tile_cols        = rt_base<T, layout, shape>::cols;
    static constexpr int base_tile_stride      = rt_base<T, layout, shape>::stride;
    static constexpr int base_tile_packed_per_stride = rt_base<T, layout, shape>::packed_per_stride;
    static constexpr int base_tile_num_strides = rt_base<T, layout, shape>::num_strides;
    static constexpr int base_tile_reductions = rt_base<T, layout, shape>::reductions;
    static constexpr int base_tile_threads_per_reduction = rt_base<T, layout, shape>::threads_per_reduction;
    static constexpr int base_tile_elements_per_stride_group = rt_base<T, layout, shape>::elements_per_stride_group;

    static constexpr int num_packed = rt_base<T, layout, shape>::num_packed;
    static constexpr int num_elements        = rt_base<T, layout, shape>::num_elements        * width * height;
    static constexpr int elements_per_thread = rt_base<T, layout, shape>::elements_per_thread * width * height;
    static constexpr int packed_per_thread   = rt_base<T, layout, shape>::packed_per_thread   * width * height;
    static constexpr int packed_per_base_tile     = rt_base<T, layout, shape>::packed_per_thread;
    static constexpr int elements_per_base_tile  = rt_base<T, layout, shape>::elements_per_thread;

    rt_base<T, layout, shape> tiles[height][width];

    using row_vec = rv<T, cols, base_tile_cols, shape, typename rt_base<T, layout, shape>::row_vec_layout>;
    using col_vec = rv<T, rows, base_tile_rows, shape, typename rt_base<T, layout, shape>::col_vec_layout>;

    using subtile = rt<T, shape::rows, shape::cols, layout, shape>;
};

namespace ducks {
namespace rt {
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

template<typename T> concept row_layout = all<T> && std::is_same_v<typename T::layout, rt_layout::row>;
template<typename T> concept col_layout = all<T> && std::is_same_v<typename T::layout, rt_layout::col>;
}
}

template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16_wmma_k16> using rt_fl = rt<float, _r, _c, layout, shape>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16_wmma_k16> using rt_bf  = rt<bf16,  _r, _c, layout, shape>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16_wmma_k16> using rt_hf  = rt<half,  _r, _c, layout, shape>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x128> using rt_fp8e4m3 = rt<fp8e4m3, _r, _c, layout, shape>;

} // namespace kittens
