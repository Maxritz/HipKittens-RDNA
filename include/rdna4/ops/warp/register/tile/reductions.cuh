/**
 * @file
 * @brief Reductions on register tiles (RDNA4 gfx1201).
 */

#pragma once

#include <type_traits>
#include <concepts>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace rt {

template<ducks::rt::all RT, ducks::rt_layout::all L=ducks::rt_layout::row>
__device__ inline auto sum(RT &t) {
    return t;
}

} // namespace rt
} // namespace kittens
