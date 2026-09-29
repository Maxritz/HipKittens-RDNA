/**
 * @file
 * @brief Shared tile shapes and swizzle maps for RDNA4 (gfx1201).
 *
 * RDNA4 has 64 LDS banks (vs CDNA5's 32+32). The XOR swizzle pattern
 * differs: banks are 32-bit wide, so a 32-bit element conflicts when
 * `addr[8:6]` differs (bits 8-6 select among 64 banks). For 4-byte types
 * the identity map is used (no swizzle needed at 4B width with 64 banks).
 * For 2-byte types, a different XOR mask is needed to avoid bank conflicts.
 *
 * The `st_16x32_padded` shape carries periodic padding to break bank
 * conflicts, same as CDNA5 but with different stride parameters tuned
 * for the 64-bank geometry.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {

namespace st_shape {

struct st_16x16 {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x16_swizzled {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2) {
            return 4;
        } else if constexpr (sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 2) {
            // RDNA4: 64 banks, 32-bit wide. XOR on bits [8:6] for bank conflict avoidance.
            // For 2-byte elements: offset granularity is 2 bytes, banks selected by bits [8:6].
            const uint32_t bank_bits = (offset >> 6) & 0x7; // bits 8:6 of byte offset -> bank id
            const uint32_t swizzle = (bank_bits >> 3) << 6;  // re-XOR pattern for 64 banks
            const uint32_t swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(_T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_32x32 {
    static constexpr int rows = 32;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 2) {
            // RDNA4: 64 banks. Two-level XOR for 32x32 at 2-byte.
            const uint32_t first_swizzle  = ((offset % 2048) >> 8) << 6;
            const uint32_t second_swizzle = ((offset % 4096) >> 9) << 5;
            const uint32_t swizzled_offset = offset ^ first_swizzle ^ second_swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(_T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x32 {
    static constexpr int rows = 16;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 2) {
            const uint32_t swizzle = ((offset % 2048) >> 8) << 6;
            const uint32_t swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(_T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_32x16 {
    static constexpr int rows = 32;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 2) {
            const uint32_t swizzle = ((offset % 2048) >> 8) << 5;
            const uint32_t swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(_T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_8x32 {
    static constexpr int rows = 8;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(_T) * (r * cols + c);

        if constexpr (sizeof(_T) == 1) {
            const uint32_t swizzle = ((offset % (16*128)) >> 8) << 4;
            const uint32_t swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

/**
 * @brief 16x32 shared subtile with LDS bank-conflict padding -- gfx1201 (RDNA4).
 *
 * RDNA4 has 64 LDS banks at 32-bit width. For 2-byte elements, padding every
 * 64 elements with 4 pad elements (8 bytes) avoids bank conflicts within a 32-lane warp.
 */
template<int PAD_INTERVAL = 64, int PAD_AMOUNT = 4>
struct st_16x32_padded {
    static constexpr int rows          = 16;
    static constexpr int cols          = 32;
    static constexpr int pad_interval  = PAD_INTERVAL;
    static constexpr int pad_amount    = PAD_AMOUNT;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const int flat = r * cols + c;
        const int padded_flat = flat + (flat / PAD_INTERVAL) * PAD_AMOUNT;
        return sizeof(_T) * padded_flat;
    }

    static constexpr int padded_elems(int total_elems) {
        return total_elems + (total_elems / PAD_INTERVAL) * PAD_AMOUNT;
    }

    __device__ __host__ __forceinline__ static constexpr int padded(int flat) {
        return flat + (flat / PAD_INTERVAL) * PAD_AMOUNT;
    }
    static constexpr int storage_elems(int total_elems) { return padded_elems(total_elems); }
};

template<typename T>
concept all = std::is_same_v<T, st_16x16> ||
              std::is_same_v<T, st_16x16_swizzled> ||
              std::is_same_v<T, st_32x32> ||
              std::is_same_v<T, st_16x32> ||
              std::is_same_v<T, st_32x16> ||
              std::is_same_v<T, st_8x32>  ||
              std::is_same_v<T, st_16x128> ||
              std::is_same_v<T, st_16x32_padded<>> ||
              std::is_same_v<T, st_16x32_padded<64, 4>> ||
              std::is_same_v<T, st_16x32_padded<64, 8>> ||
              std::is_same_v<T, st_16x32_padded<128, 4>> ||
              std::is_same_v<T, st_16x32_padded<128, 8>> ||
              std::is_same_v<T, st_16x32_padded<256, 4>>;

template<typename S>
concept padded = requires {
    { S::pad_interval } -> std::convertible_to<int>;
    { S::pad_amount   } -> std::convertible_to<int>;
    S::padded(0);
    S::storage_elems(0);
};

template<typename S, typename T>
__host__ __device__ constexpr bool permutes_for() {
    for (int r = 0; r < S::rows; r++)
        for (int c = 0; c < S::cols; c++)
            if (S::template swizzle<T>(int2{r, c})
                != static_cast<uint32_t>(sizeof(T) * (r * S::cols + c)))
                return true;
    return false;
}

template<typename S, typename T>
concept swizzled = !padded<S> && permutes_for<S, T>();

template<typename S, int E>
__host__ __device__ constexpr bool run_is_contiguous() {
    if constexpr (padded<S>) return (S::pad_interval % E == 0) && (S::pad_amount % E == 0);
    else return true;
}

} // namespace st_shape
} // namespace ducks
} // namespace kittens
