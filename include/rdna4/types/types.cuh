/**
 * @file
 * @brief An aggregate header file for all the register and shared types (RDNA4).
 */

#pragma once

#include "register/register.cuh"
#include "shared/shared.cuh"
#include "global/global.cuh"

namespace kittens {

template<typename T>
using row_vec = T::row_vec;

template<typename T>
using col_vec = T::col_vec;

using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;

using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

using rt_16x16_s = ducks::rt_shape::rt_16x16;
using rt_32x32_s = ducks::rt_shape::rt_32x32;
using rt_32x32_8_s = ducks::rt_shape::rt_32x32_8;
using rt_16x32_s = ducks::rt_shape::rt_16x32;
using rt_32x16_s = ducks::rt_shape::rt_32x16;
using rt_32x16_4_s = ducks::rt_shape::rt_32x16_4;
using rt_16x32_4_s = ducks::rt_shape::rt_16x32_4;
using rt_16x128_s = ducks::rt_shape::rt_16x128;

using st_16x16_s = ducks::st_shape::st_16x16;
using st_16x16_swizzled_s = ducks::st_shape::st_16x16_swizzled;
using st_32x32_s = ducks::st_shape::st_32x32;
using st_16x32_s = ducks::st_shape::st_16x32;
using st_32x16_s = ducks::st_shape::st_32x16;
using st_8x32_s = ducks::st_shape::st_8x32;
using st_16x128_s = ducks::st_shape::st_16x128;

}
