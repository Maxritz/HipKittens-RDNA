/**
 * @file
 * @brief Templated layouts for global memory (RDNA4 gfx1201).
 *
 * Same as CDNA5. Global memory descriptors are layout-aware.
 */

#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"
#include "gl_layout.cuh"

namespace kittens {

namespace ducks {
namespace gl {
struct identifier {};
}
}

template<typename _T, int b, int d, int r, int c,
         ducks::gl_layout::all _Layout = ducks::gl_layout::row_major>
struct gl {
    using identifier = ducks::gl::identifier;

    using layout = _Layout;
    using T     = base_types::packing<_T>::unpacked_type;
    using T2    = base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;

    static constexpr int __b__ = b, __d__ = d, __r__ = r, __c__ = c;

    ducks::gl::make_dim_t<b> batch_internal;
    ducks::gl::make_dim_t<d> depth_internal;
    ducks::gl::make_dim_t<r> rows_internal;
    ducks::gl::make_dim_t<c> cols_internal;

    template <int B=__b__> __device__ __host__ static constexpr std::enable_if_t<(B > 0), int> batch() { return B; }
    template <int B=__b__> __device__ __host__ std::enable_if_t<(B == -1), int> batch() const { return batch_internal; }
    template <int D=__d__> __device__ __host__ static constexpr std::enable_if_t<(D > 0), int> depth() { return D; }
    template <int D=__d__> __device__ __host__ std::enable_if_t<(D == -1), int> depth() const { return depth_internal; }
    template <int R=__r__> __device__ __host__ static constexpr std::enable_if_t<(R > 0), int> rows() { return R; }
    template <int R=__r__> __device__ __host__ std::enable_if_t<(R == -1), int> rows() const { return rows_internal; }
    template <int C=__c__> __device__ __host__ static constexpr std::enable_if_t<(C > 0), int> cols() { return C; }
    template <int C=__c__> __device__ __host__ std::enable_if_t<(C == -1), int> cols() const { return cols_internal; }

    __host__ inline gl(T *_data,
                         ducks::gl::make_arg_t<b> _batch,
                         ducks::gl::make_arg_t<d> _depth,
                         ducks::gl::make_arg_t<r> _rows,
                         ducks::gl::make_arg_t<c> _cols) :
            raw_ptr(_data), batch_internal(_batch), depth_internal(_depth), rows_internal(_rows), cols_internal(_cols) {}
    __host__ __device__ inline gl(const gl &other) :
            raw_ptr(other.raw_ptr), batch_internal(other.batch_internal), depth_internal(other.depth_internal), rows_internal(other.rows_internal), cols_internal(other.cols_internal) {}
    __device__ inline T& operator[](const coord<ducks::default_type> &idx) const {
        return raw_ptr[this->idx(idx)];
    }
    __device__ inline int64_t idx(const coord<ducks::default_type> &idx) const
        requires std::is_same_v<layout, ducks::gl_layout::row_major> {
        return ((int64_t(idx.b)*depth() + idx.d)*rows() + idx.r)*cols() + idx.c;
    }
    __device__ inline int64_t idx(const coord<ducks::default_type> &idx) const
        requires std::is_same_v<layout, ducks::gl_layout::col_major> {
        return ((int64_t(idx.b)*depth() + idx.d)*cols() + idx.c)*rows() + idx.r;
    }
    template<int axis> __device__ inline size_t shape() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return size_t(batch()); }
        else if constexpr (axis==1) { return size_t(depth()); }
        else if constexpr (axis==2) { return size_t(rows()); }
        else                          { return size_t(cols()); }
    }
    template<int axis> __device__ inline size_t stride() const
        requires std::is_same_v<layout, ducks::gl_layout::row_major> {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return depth()*rows()*cols(); }
        else if constexpr (axis==1) { return rows()*cols(); }
        else if constexpr (axis==2) { return cols(); }
        else                        { return 1; }
    }
    template<int axis> __device__ inline size_t stride() const
        requires std::is_same_v<layout, ducks::gl_layout::col_major> {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return depth()*rows()*cols(); }
        else if constexpr (axis==1) { return rows()*cols(); }
        else if constexpr (axis==2) { return 1; }
        else                        { return rows(); }
    }
    template<int axis> __device__ inline size_t offset(int row, int col) const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return (size_t(row)*depth() + size_t(col)) * rows()*cols(); }
        else if constexpr (axis==1) { return (size_t(row)*cols() + size_t(col)) * rows(); }
        else if constexpr (axis==2) { return size_t(row)*cols() + col; }
        else                        { return size_t(row)*rows() + col; }
    }
    template<int axis> __device__ inline size_t offset(int row) const {
        return offset<axis>(row, 0);
    }
    template<int axis> __device__ inline size_t offset() const {
        return offset<axis>(0, 0);
    }
    template<int axis> __device__ inline int64_t offset(int row, int col) const
        requires std::is_same_v<layout, ducks::gl_layout::col_major> {
        return offset<axis>(row, col);
    }
};

namespace ducks {
namespace gl {

template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

template<typename T> concept row_layout = all<T> && std::is_same_v<typename T::layout, gl_layout::row_major>;
template<typename T> concept col_layout = all<T> && std::is_same_v<typename T::layout, gl_layout::col_major>;

}
}

template<int _r, int _c, ducks::gl_layout::all layout=ducks::gl_layout::row_major> using gl_fl = gl<float,  1, 1, _r, _c, layout>;
template<int _r, int _c, ducks::gl_layout::all layout=ducks::gl_layout::row_major> using gl_bf = gl<bf16,  1, 1, _r, _c, layout>;
template<int _r, int _c, ducks::gl_layout::all layout=ducks::gl_layout::row_major> using gl_hf = gl<half,  1, 1, _r, _c, layout>;
template<int _r, int _c, ducks::gl_layout::all layout=ducks::gl_layout::row_major> using gl_fp8e4m3 = gl<fp8e4m3, 1, 1, _r, _c, layout>;

} // namespace kittens
