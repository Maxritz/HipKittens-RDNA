/**
 * @file
 * @brief rv_layout for register vectors (RDNA4 gfx1201).
 *
 * Same layout concepts as CDNA5 for vectors.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
namespace rv_layout {

struct naive {};
struct align {};
struct ortho {};

template<typename T>
concept all = std::is_same_v<T, naive> || std::is_same_v<T, align> || std::is_same_v<T, ortho>;

} // namespace rv_layout
} // namespace ducks
} // namespace kittens
