/**
 * @file
 * @brief Layout concepts for register tiles (RDNA4 gfx1201).
 *
 * Same as CDNA5.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
namespace rt_layout {

struct row {};
struct col {};

template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

} // namespace rt_layout
} // namespace ducks
} // namespace kittens
