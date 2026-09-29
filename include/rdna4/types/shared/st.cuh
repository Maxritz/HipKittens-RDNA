/**
 * @file
 * @brief The ThunderKittens shared tile struct (RDNA4 gfx1201).
 *
 * Mirrors CDNA5's st but with RDNA4-specific swizzle patterns (64 LDS banks).
 * The default shape is st_16x32_padded to carry periodic bank-conflict padding.
 */

#pragma once

#include "../../common/common.cuh"
#include "sv.cuh"
#include "st_shape.cuh"
#include "st_layout.cuh"

namespace kittens {
namespace ducks {
namespace st {
struct identifier {};
}
}

template<
    typename _T,
    int _rows,
    int _cols,
    ducks::st_shape::all _shape,
    ducks::st_layout::all _layout = ducks::st_layout::row
>
struct KITTENS_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T;
    using shape = _shape;
    using layout = _layout;

    static constexpr int underlying_rows              = _rows;
    static constexpr int underlying_cols              = _cols;
    static constexpr int underlying_num_elements      = underlying_rows * underlying_cols;

    static constexpr int underlying_subtile_rows      = shape::rows;
    static constexpr int underlying_subtile_cols      = shape::cols;
    static constexpr int underlying_subtile_row_bytes = shape::cols * sizeof(T);
    static constexpr int underlying_subtile_elements  = underlying_subtile_rows * underlying_subtile_cols;
    static constexpr int underlying_subtile_bytes     = underlying_subtile_elements * sizeof(T);
    static constexpr int underlying_subtile_bytes_per_thread = shape::template bytes_per_thread<T>();

    static constexpr int underlying_subtiles_per_row  = underlying_cols / underlying_subtile_cols;
    static constexpr int underlying_subtiles_per_col  = underlying_rows / underlying_subtile_rows;

    static constexpr int rows                = _rows;
    static constexpr int cols                = _cols;
    static constexpr int num_elements        = rows * cols;

    static constexpr int subtiles_per_row    = cols / underlying_subtile_cols;
    static constexpr int subtiles_per_col    = rows / underlying_subtile_rows;

    static_assert(base_types::packing<dtype>::num() == 1, "must be a 1-packed type");

    template<typename S> static constexpr int storage_for() {
        if constexpr (ducks::st_shape::padded<S>) return S::storage_elems(rows*cols);
        else return rows*cols;
    }
    static constexpr int storage_elements = storage_for<shape>();

    dtype data[storage_elements];

    __device__ __host__ __forceinline__ static constexpr int idx(int r, int c) {
        if constexpr (ducks::st_shape::padded<shape>) {
            return shape::padded(layout_flat(r, c));
        } else {
            constexpr int SR = shape::rows, SC = shape::cols;
            const int sub_id = (r / SR) * (cols / SC) + (c / SC);
            return sub_id * (SR * SC)
                 + int(shape::template swizzle<T>({r % SR, c % SC})) / int(sizeof(T));
        }
    }
    __device__ __forceinline__ static T* idx(T *ptr, int2 coord) {
        return &ptr[idx(coord.x, coord.y)];
    }
    __device__ __forceinline__ static uint32_t idx(uint32_t ptr, int2 coord) {
        return ptr + sizeof(T) * idx(coord.x, coord.y);
    }

    __device__ __forceinline__ dtype& operator[](const int2 &rowcol) {
        return data[idx(rowcol.x, rowcol.y)];
    }
    __device__ __forceinline__ const dtype& operator[](const int2 &rowcol) const {
        return data[idx(rowcol.x, rowcol.y)];
    }
    __device__ __forceinline__ dtype& operator[](int i)             { return data[i]; }
    __device__ __forceinline__ const dtype& operator[](int i) const { return data[i]; }

    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        return shape::template swizzle<T>(coord);
    }

private:
    __device__ __host__ __forceinline__ static constexpr int layout_flat(int r, int c) {
        if constexpr (std::is_same_v<layout, ducks::st_layout::col>) return c * rows + r;
        else                                                        return r * cols + c;
    }

public:
    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;

    template<int subtile_rows, int subtile_cols> using subtile = st_subtile<st<_T, _rows, _cols, _shape, _layout>, subtile_rows, subtile_cols>;
};

template<
    typename _ST,
    int _subtile_rows,
    int _subtile_cols
>
struct st_subtile {
    using identifier = ducks::st::identifier;
    using ST = _ST;
    using T = ST::T;
    using T2 = ST::T2;
    using dtype = T;
    using shape = ST::shape;
    using layout = ST::layout;

    static constexpr int underlying_rows              = ST::underlying_rows;
    static constexpr int underlying_cols              = ST::underlying_cols;
    static constexpr int underlying_num_elements      = ST::underlying_num_elements;
    static constexpr int underlying_subtile_cols      = ST::underlying_subtile_cols;
    static constexpr int underlying_subtile_row_bytes = ST::underlying_subtile_row_bytes;
    static constexpr int underlying_subtile_rows      = ST::underlying_subtile_rows;
    static constexpr int underlying_subtile_elements  = ST::underlying_subtile_elements;
    static constexpr int underlying_subtile_bytes     = ST::underlying_subtile_bytes;
    static constexpr int underlying_subtile_bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    static constexpr int underlying_subtiles_per_row  = ST::underlying_subtiles_per_row;
    static constexpr int underlying_subtiles_per_col  = ST::underlying_subtiles_per_col;

    static constexpr int rows                = _subtile_rows;
    static constexpr int cols                = _subtile_cols;
    static constexpr int num_elements        = rows * cols;
    static constexpr int subtiles_per_row    = cols / underlying_subtile_cols;
    static constexpr int subtiles_per_col    = rows / underlying_subtile_rows;

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(ST &src, int2 rowcol) {
        row_offset = rowcol.x * rows;
        col_offset = rowcol.y * cols;
        const int subtile_row_offset = row_offset / underlying_subtile_rows;
        const int subtile_col_offset = col_offset / underlying_subtile_cols;
        const int subtile_id = subtile_row_offset * underlying_subtiles_per_row + subtile_col_offset;
        const int subtile_offset = subtile_id * underlying_subtile_elements;
        data = &src.data[subtile_offset];
    }

    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        return ST::swizzle(coord);
    }

    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;
};

namespace ducks {
namespace st {

template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

template<typename T> concept row_layout = all<T> && std::is_same_v<typename T::layout, st_layout::row>;
template<typename T> concept col_layout = all<T> && std::is_same_v<typename T::layout, st_layout::col>;

}
}

template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_bf = st<bf16,  _height, _width, _shape, _layout>;
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_hf = st<half,  _height, _width, _shape, _layout>;
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_fl = st<float,  _height, _width, _shape, _layout>;
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_fp8e4m3 = st<fp8e4m3, _height, _width, _shape, _layout>;
