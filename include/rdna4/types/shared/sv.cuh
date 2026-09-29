/**
 * @file
 * @brief The ThunderKittens shared vector struct (RDNA4 gfx1201).
 *
 * Same as CDNA5: uniform linear layout in shared memory.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

namespace kittens {

namespace ducks {
namespace sv {
struct identifier {};
}
}

template<typename _T, size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {
    using identifier = ducks::sv::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T;

    static constexpr int length = _length;
    static constexpr int num_alloc_elements = length;

    dtype data[num_alloc_elements];

    __device__ static inline T* idx(T *ptr, int idx) {
        return ptr[idx];
    }

    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }

    template<size_t sub_length> using subvec = sv<dtype, sub_length>;
};

namespace ducks {
namespace sv {

template<typename T>
concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

}
}

template<size_t _length> using sv_bf = sv<bf16,  _length>;
template<size_t _length> using sv_hf = sv<half,  _length>;
template<size_t _length> using sv_fl = sv<float, _length>;
template<size_t _length> using sv_fp8e4m3 = sv<fp8e4m3, _length>;

} // namespace kittens
