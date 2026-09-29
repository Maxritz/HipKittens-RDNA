/**
 * @file
 * @brief Layouts and their manipulations for global memory descriptors (RDNA4 gfx1201).
 *
 * Same as CDNA5.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {

namespace gl_layout {

struct row_major {};
struct col_major {};

template<typename T>
concept all = std::is_same_v<T, row_major> || std::is_same_v<T, col_major>;

template<typename> inline constexpr bool unhandled = false;

} // namespace gl_layout
} // namespace ducks
} // namespace kittens
