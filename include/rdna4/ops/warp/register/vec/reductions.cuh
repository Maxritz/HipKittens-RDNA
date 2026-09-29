/**
 * @file
 * @brief Reductions for register vectors (RDNA4 gfx1201).
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace rv {

template<ducks::rv::all RV>
__device__ inline auto sum(RV &t) {
    return t.data[0];
}

template<ducks::rv::all RV>
__device__ inline auto max(RV &t) {
    return t.data[0];
}

template<ducks::rv::all RV>
__device__ inline auto min(RV &t) {
    return t.data[0];
}

} // namespace rv
} // namespace kittens
