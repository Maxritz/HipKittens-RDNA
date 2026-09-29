/**
 * @file
 * @brief Layout concepts for shared memory tiles (RDNA4 gfx1201).
 *
 * Same layout concept as CDNA5 (row/col). On RDNA4 the bank-conflict
 * avoidance is handled differently (64 LDS banks vs CDNA5's 32), but
 * the layout type system is identical.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {

namespace st_layout {

struct row {};
struct col {};

template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

} // namespace st_layout
} // namespace ducks
} // namespace kittens
