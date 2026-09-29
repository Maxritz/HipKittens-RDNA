/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace rt_shape
* 
* @brief A namespace for template metaprogramming with register tile layouts.
* Assumption below is that the col is the reduction dimension
*/
namespace rt_shape {
  
template<int _rows, int _cols, int _stride>
struct rt_shape {
    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
    static constexpr int stride = _stride;
    static constexpr int num_elements = rows*cols;
    static constexpr int elements_per_thread = num_elements / kittens::WARP_THREADS;
    static constexpr int num_strides = elements_per_thread / stride;
};

// WMMA shapes: 16x16 K=32 (WMMA_F32_16X16X32_BF16/F16) and 16x32/32x16 for operands
using rt_16x16 = rt_shape<16, 16, 4>;
using rt_32x32 = rt_shape<32, 32, 4>;
using rt_32x32_8 = rt_shape<32, 32, 8>;
using rt_16x32 = rt_shape<16, 32, 8>;
using rt_32x16 = rt_shape<32, 16, 8>;
using rt_32x16_4 = rt_shape<32, 16, 4>;
using rt_16x32_4 = rt_shape<16, 32, 4>;
using rt_16x128 = rt_shape<16, 128, 16>;

// WMMA K=32 operand shapes: A is 16x32, B is 32x16 (matching WMMA 16x16x32 layout)
using rt_16x32_wmma = rt_shape<16, 32, 8>;
using rt_32x16_wmma = rt_shape<32, 16, 8>;

template<typename T>
concept all = std::is_same_v<T, rt_16x16> ||
              std::is_same_v<T, rt_32x32> ||
              std::is_same_v<T, rt_32x32_8> ||
              std::is_same_v<T, rt_16x32> ||
              std::is_same_v<T, rt_32x16> ||
              std::is_same_v<T, rt_32x16_4> ||
              std::is_same_v<T, rt_16x32_4> ||
              std::is_same_v<T, rt_16x128> ||
              std::is_same_v<T, rt_16x32_wmma> ||
              std::is_same_v<T, rt_32x16_wmma>;

/**
 * @brief A struct to generate a transposed layout.
 * Note: on RDNA4, the accumulator layout becomes the col layout when transposed.
 */
  template<all L> struct transpose      { using type = rt_16x16; };
  template<>      struct transpose<rt_32x32> { using type = rt_32x32; };
  template<>      struct transpose<rt_32x32_8> { using type = rt_32x32_8; };
  template<>      struct transpose<rt_16x32> { using type = rt_32x16; };
  template<>      struct transpose<rt_32x16> { using type = rt_16x32; };
  template<>      struct transpose<rt_32x16_4> { using type = rt_16x32_4; };
  template<>      struct transpose<rt_16x32_4> { using type = rt_32x16_4; };
} // namespace rt_shape
} // namespace ducks
} // namespace kittens
