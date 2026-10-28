#include <cstdint>
#include <cstddef>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <immintrin.h>

static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -14;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
            mant &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline int32_t hsum256_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// Quantize activations to signed int16 in small groups: 32 values for Q4_K /
// Q5_K and 16 for Q6_K. The 15-bit range keeps error tiny while allowing AVX2
// integer dot products to replace repeated int-to-float conversions and FMAs.
// Layout: x[cols], optional 32-value sums, one float scale per quant group,
// followed by cols int16 activation values in the final float slots.
static std::vector<float> quantize_input(const float* x, int cols, int sum_groups, int quant_group_size) {
    const int quant_groups = cols / quant_group_size;
    const size_t float_count = static_cast<size_t>(cols) + sum_groups + quant_groups + cols / 2;
    std::vector<float> packed(float_count);
    std::memcpy(packed.data(), x, static_cast<size_t>(cols) * sizeof(float));
    for (int g = 0; g < sum_groups; ++g) {
        float sum = 0.0f;
        const int first = g * 32;
        for (int j = 0; j < 32; ++j) sum += x[first + j];
        packed[cols + g] = sum;
    }
    float* scales = packed.data() + cols + sum_groups;
    auto* qx = reinterpret_cast<int16_t*>(scales + quant_groups);
    for (int g = 0; g < quant_groups; ++g) {
        float max_abs = 0.0f;
        const int first = g * quant_group_size;
        for (int j = 0; j < quant_group_size; ++j) max_abs = std::max(max_abs, std::fabs(x[first + j]));
        const float scale = max_abs > 1.0e-20f ? max_abs / 32767.0f : 1.0f;
        scales[g] = scale;
        for (int j = 0; j < quant_group_size; ++j) {
            long q = std::lrintf(x[first + j] / scale);
            q = std::max(-32767L, std::min(32767L, q));
            qx[first + j] = static_cast<int16_t>(q);
        }
    }
    return packed;
}

// Q4_K/Q5_K use one activation scale per 32-value group. An int8 activation
// vector lets the AVX2 kernels consume the packed weight nibbles directly with
// VPMADDUBSW instead of expanding every nibble to int16 first.
static std::vector<float> quantize_input_q8(const float* x, int cols, int sum_groups) {
    const int quant_groups = cols / 32;
    const size_t float_count = static_cast<size_t>(cols) + sum_groups + quant_groups + (cols + 3) / 4;
    std::vector<float> packed(float_count);
    std::memcpy(packed.data(), x, static_cast<size_t>(cols) * sizeof(float));
    for (int g = 0; g < sum_groups; ++g) {
        float sum = 0.0f;
        for (int j = 0; j < 32; ++j) sum += x[g * 32 + j];
        packed[cols + g] = sum;
    }
    float* scales = packed.data() + cols + sum_groups;
    auto* qx = reinterpret_cast<int8_t*>(scales + quant_groups);
    for (int g = 0; g < quant_groups; ++g) {
        const int first = g * 32;
        float max_abs = 0.0f;
        for (int j = 0; j < 32; ++j) max_abs = std::max(max_abs, std::fabs(x[first + j]));
        const float scale = max_abs > 1.0e-20f ? max_abs / 127.0f : 1.0f;
        scales[g] = scale;
        for (int j = 0; j < 32; ++j) {
            long q = std::lrintf(x[first + j] / scale);
            q = std::max(-127L, std::min(127L, q));
            qx[first + j] = static_cast<int8_t>(q);
        }
    }
    return packed;
}

static inline __m256i dot_u8_s8_32_lanes(__m256i q8, const int8_t* x8) {
    const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x8));
    const __m256i pairs = _mm256_maddubs_epi16(q8, xv);
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

static inline int q4_scale(const uint8_t* s, int g) {
    if (g < 4) return s[g] & 0x3f;
    return (s[g + 4] & 0x0f) | ((s[g - 4] >> 6) << 4);
}
static inline int q4_min(const uint8_t* s, int g) {
    if (g < 4) return s[g + 4] & 0x3f;
    return (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
}

static float q8_0_row(const uint8_t* row, const float* x, int cols) {
    // Each 8-value slice has its own accumulator across all blocks. Explicitly
    // unroll the four slices so the compiler can overlap independent loads and
    // FMAs without a per-block horizontal reduction or branch-heavy dispatch.
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const int blocks = cols / 32;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* p = row + b * 34;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
        const __m256 scale = _mm256_set1_ps(d);
        const __m128i p0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q));
        const __m128i p1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + 8));
        const __m128i p2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + 16));
        const __m128i p3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + 24));
        const __m256 qf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p0)), scale);
        const __m256 qf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p1)), scale);
        const __m256 qf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p2)), scale);
        const __m256 qf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p3)), scale);
        acc0 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(x + b * 32), acc0);
        acc1 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(x + b * 32 + 8), acc1);
        acc2 = _mm256_fmadd_ps(qf2, _mm256_loadu_ps(x + b * 32 + 16), acc2);
        acc3 = _mm256_fmadd_ps(qf3, _mm256_loadu_ps(x + b * 32 + 24), acc3);
    }
    return (hsum256(acc0) + hsum256(acc1)) + (hsum256(acc2) + hsum256(acc3));
}

// Q4_0 is common in compact GGUF checkpoints but was previously falling back
// to full row dequantization on every token. Quantize activations once per
// 32-value block and do the signed 4-bit dot in AVX2 integer lanes.
static float q4_0_row(const uint8_t* row, const float* x, int cols) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    const int blocks = cols / 32;
    const float* activation_scales = x + cols + blocks;
    const auto* qx = reinterpret_cast<const int16_t*>(activation_scales + blocks);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m256i zero_point = _mm256_set1_epi16(8);
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* p = row + b * 18;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2));
        const __m256i low = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_and_si128(packed, nibble_mask)), zero_point);
        const __m256i high = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask)), zero_point);
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qx + b * 32));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qx + b * 32 + 16));
        const __m256i dot0 = _mm256_madd_epi16(low, x0);
        const __m256i dot1 = _mm256_madd_epi16(high, x1);
        const __m256 factor = _mm256_set1_ps(d * activation_scales[b]);
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), factor, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot1), factor, acc1);
    }
    // The SIMD lanes already hold signed values (nibble - 8), so no separate
    // zero-point correction term is required.
    return hsum256(acc0) + hsum256(acc1);
}

static float q4k_row(const uint8_t* row, const float* x, int cols) {
    // Keep dot products in SIMD lanes across blocks. The old path horizontally
    // reduced every 32-value group to a scalar, creating many reductions and a
    // long dependency chain. Four independent vector accumulators reduce only
    // once per stream at the end of the row.
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    float minimum_total = 0.0f;
    const int blocks = cols / 256;
    const int groups32 = blocks * 8;
    const float* activation_scales = x + cols + groups32;
    const auto* qx = reinterpret_cast<const int8_t*>(activation_scales + groups32);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* p = row + b * 144;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const float dm = half_to_float(*reinterpret_cast<const uint16_t*>(p + 2));
        const uint8_t* scales = p + 4;
        const uint8_t* qs = p + 16;
        auto accumulate_pair = [&](int g, __m256& acc) {
            const float scale0 = d * q4_scale(scales, g);
            const float minimum0 = dm * q4_min(scales, g);
            const float scale1 = d * q4_scale(scales, g + 1);
            const float minimum1 = dm * q4_min(scales, g + 1);
            const int qbase = (g / 2) * 32;
            const int xbase0 = b * 256 + g * 32;
            const int xbase1 = xbase0 + 32;
            // Each byte stores one low-nibble value for group g and one
            // high-nibble value for group g+1, so both 32-value dots are
            // computed directly without expanding to int16.
            const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + qbase));
            const __m256i q0 = _mm256_and_si256(bytes, nibble_mask);
            const __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nibble_mask);
            const __m256i dot0 = dot_u8_s8_32_lanes(q0, qx + xbase0);
            const __m256i dot1 = dot_u8_s8_32_lanes(q1, qx + xbase1);
            const float sx0 = activation_scales[b * 8 + g];
            const float sx1 = activation_scales[b * 8 + g + 1];
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), _mm256_set1_ps(scale0 * sx0), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot1), _mm256_set1_ps(scale1 * sx1), acc);
            minimum_total += minimum0 * x[cols + b * 8 + g];
            minimum_total += minimum1 * x[cols + b * 8 + g + 1];
        };
        accumulate_pair(0, acc0);
        accumulate_pair(2, acc1);
        accumulate_pair(4, acc2);
        accumulate_pair(6, acc3);
    }
    return ((hsum256(acc0) + hsum256(acc1)) + (hsum256(acc2) + hsum256(acc3))) - minimum_total;
}

static float q5k_row(const uint8_t* row, const float* x, int cols) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    float minimum_total = 0.0f;
    const int blocks = cols / 256;
    const int groups32 = blocks * 8;
    const float* activation_scales = x + cols + groups32;
    const auto* qx = reinterpret_cast<const int8_t*>(activation_scales + groups32);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i bit_mask = _mm256_set1_epi8(0x01);
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* p = row + b * 176;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const float dm = half_to_float(*reinterpret_cast<const uint16_t*>(p + 2));
        const uint8_t* scales = p + 4;
        const uint8_t* qh = p + 16;
        const uint8_t* qs = p + 48;
        auto accumulate_pair = [&](int g, __m256& acc) {
            const float scale0 = d * q4_scale(scales, g), minimum0 = dm * q4_min(scales, g);
            const float scale1 = d * q4_scale(scales, g + 1), minimum1 = dm * q4_min(scales, g + 1);
            const int qbase = (g / 2) * 32;
            const int xbase0 = b * 256 + g * 32, xbase1 = xbase0 + 32;
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + qbase));
            const __m256i low0 = _mm256_and_si256(packed, nibble_mask);
            const __m256i low1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
            const __m256i high_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            const __m256i high0 = _mm256_and_si256(_mm256_srli_epi16(high_packed, g), bit_mask);
            const __m256i high1 = _mm256_and_si256(_mm256_srli_epi16(high_packed, g + 1), bit_mask);
            const __m256i q0 = _mm256_or_si256(low0, _mm256_slli_epi16(high0, 4));
            const __m256i q1 = _mm256_or_si256(low1, _mm256_slli_epi16(high1, 4));
            const __m256i dot0 = dot_u8_s8_32_lanes(q0, qx + xbase0);
            const __m256i dot1 = dot_u8_s8_32_lanes(q1, qx + xbase1);
            const float sx0 = activation_scales[b * 8 + g];
            const float sx1 = activation_scales[b * 8 + g + 1];
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), _mm256_set1_ps(scale0 * sx0), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot1), _mm256_set1_ps(scale1 * sx1), acc);
            minimum_total += minimum0 * x[cols + b * 8 + g];
            minimum_total += minimum1 * x[cols + b * 8 + g + 1];
        };
        accumulate_pair(0, acc0);
        accumulate_pair(2, acc1);
        accumulate_pair(4, acc2);
        accumulate_pair(6, acc3);
    }
    return ((hsum256(acc0) + hsum256(acc1)) + (hsum256(acc2) + hsum256(acc3))) - minimum_total;
}

static float q6k_row(const uint8_t* row, const float* x, int cols) {
    // Q6_K has sixteen small dot products per 256-value block. Keep their
    // eight-lane int32 results in four independent float SIMD accumulators;
    // only horizontally reduce after the whole row instead of every 16 values.
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const int blocks = cols / 256;
    const int quant_groups = cols / 16;
    const float* activation_scales = x + cols;
    const auto* qx = reinterpret_cast<const int16_t*>(activation_scales + quant_groups);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i two_bit_mask = _mm_set1_epi8(0x03);
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* p = row + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(p + 192);
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p + 208));
        auto accumulate_quad = [&](int half, int quad, __m256& acc) {
            const int pair = quad + 2;
            const int qlbase = half * 64;
            const int qhbase = half * 32;
            const int scalebase = half * 8;
            const int xbase = b * 256 + half * 128;
            const int qb = qlbase + (quad == 1 ? 32 : 0);
            // Q6_K packs quads 0/2 and 1/3 into the low/high nibbles of
            // the same bytes. Reuse each packed slice for both dot products.
            for (int j = 0; j < 32; j += 16) {
                const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + qb + j));
                const __m128i low_values = _mm_and_si128(packed, nibble_mask);
                const __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                const __m128i high_packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + qhbase + j));
                const __m128i high0 = _mm_and_si128(_mm_srli_epi16(high_packed, quad * 2), two_bit_mask);
                const __m128i high2 = _mm_and_si128(_mm_srli_epi16(high_packed, pair * 2), two_bit_mask);
                // Reconstruct unsigned 6-bit values in byte lanes before widening.
                const __m128i q0_bytes = _mm_or_si128(low_values, _mm_slli_epi16(high0, 4));
                const __m128i q2_bytes = _mm_or_si128(high_nibbles, _mm_slli_epi16(high2, 4));
                const __m256i q0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q0_bytes), _mm256_set1_epi16(32));
                const __m256i q2 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q2_bytes), _mm256_set1_epi16(32));
                const __m256i xv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qx + xbase + quad * 32 + j));
                const __m256i xv2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qx + xbase + pair * 32 + j));
                const int scale0 = scales[scalebase + quad * 2 + j / 16];
                const int scale2 = scales[scalebase + pair * 2 + j / 16];
                const __m256i dot0 = _mm256_madd_epi16(q0, xv0);
                const __m256i dot2 = _mm256_madd_epi16(q2, xv2);
                const float factor0 = d * static_cast<float>(scale0) * activation_scales[(xbase + quad * 32 + j) / 16];
                const float factor2 = d * static_cast<float>(scale2) * activation_scales[(xbase + pair * 32 + j) / 16];
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), _mm256_set1_ps(factor0), acc);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot2), _mm256_set1_ps(factor2), acc);
            }
        };
        accumulate_quad(0, 0, acc0);
        accumulate_quad(0, 1, acc1);
        accumulate_quad(1, 0, acc2);
        accumulate_quad(1, 1, acc3);
    }
    return (hsum256(acc0) + hsum256(acc1)) + (hsum256(acc2) + hsum256(acc3));
}

class MatvecPool {
public:
    using RowFn = float (*)(const uint8_t*, const float*, int);

    explicit MatvecPool(int max_threads) : max_threads_(std::max(1, max_threads)) {
        workers_.reserve(static_cast<size_t>(max_threads_ - 1));
        for (int id = 1; id < max_threads_; ++id)
            workers_.emplace_back([this, id] { worker_loop(id); });
    }

    ~MatvecPool() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }

    void run(const uint8_t* raw, const float* x, float* out, int rows, int cols,
             int bytes_per_row, int threads, RowFn fn) {
        // A GGUF inference session is sequential, but serialize native calls so
        // future GUI/background callers cannot overwrite an active job.
        std::lock_guard<std::mutex> serial(run_mutex_);
        threads = std::max(1, std::min(std::min(threads, rows), max_threads_));
        if (threads == 1 || rows < 256) {
            work_rows(raw, x, out, 0, rows, cols, bytes_per_row, fn);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            raw_ = raw; x_ = x; out_ = out; rows_ = rows; cols_ = cols;
            bytes_per_row_ = bytes_per_row; fn_ = fn;
            active_threads_ = threads;
            remaining_ = threads - 1;
            ++generation_;
        }
        start_cv_.notify_all();
        work_rows(raw, x, out, 0, rows / threads, cols, bytes_per_row, fn);
        std::unique_lock<std::mutex> lock(state_mutex_);
        done_cv_.wait(lock, [this] { return remaining_ == 0; });
    }

private:
    static void work_rows(const uint8_t* raw, const float* x, float* out,
                          int begin, int end, int cols, int bytes_per_row, RowFn fn) {
        for (int r = begin; r < end; ++r)
            out[r] = fn(raw + static_cast<size_t>(r) * bytes_per_row, x, cols);
    }

    void worker_loop(int id) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            start_cv_.wait(lock, [this, &seen_generation] {
                return stopping_ || generation_ != seen_generation;
            });
            if (stopping_) return;
            seen_generation = generation_;
            if (id >= active_threads_) continue;
            const auto raw = raw_; const auto x = x_; auto out = out_;
            const int rows = rows_, cols = cols_, bytes_per_row = bytes_per_row_;
            const int threads = active_threads_; const auto fn = fn_;
            lock.unlock();

            const int begin = static_cast<int>((static_cast<long long>(rows) * id) / threads);
            const int end = static_cast<int>((static_cast<long long>(rows) * (id + 1)) / threads);
            work_rows(raw, x, out, begin, end, cols, bytes_per_row, fn);

            lock.lock();
            if (--remaining_ == 0) done_cv_.notify_one();
        }
    }

    const int max_threads_;
    std::vector<std::thread> workers_;
    std::mutex run_mutex_, state_mutex_;
    std::condition_variable start_cv_, done_cv_;
    bool stopping_ = false;
    uint64_t generation_ = 0;
    int active_threads_ = 1, remaining_ = 0;
    const uint8_t* raw_ = nullptr;
    const float* x_ = nullptr;
    float* out_ = nullptr;
    int rows_ = 0, cols_ = 0, bytes_per_row_ = 0;
    RowFn fn_ = nullptr;
};

static void parallel_matvec(const uint8_t* raw, const float* x, float* out,
                            int rows, int cols, int bytes_per_row, int threads,
                            MatvecPool::RowFn fn) {
    threads = std::max(1, std::min(threads, rows));
    if (rows < 256) threads = 1;
    if (threads == 1) {
        for (int r = 0; r < rows; ++r)
            out[r] = fn(raw + static_cast<size_t>(r) * bytes_per_row, x, cols);
        return;
    }
    static MatvecPool pool(static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency()))));
    pool.run(raw, x, out, rows, cols, bytes_per_row, threads, fn);
}

using MatmulRowFn = void (*)(const uint8_t*, const float*, float*, int, int, float*);

class MatmulPool {
public:
    explicit MatmulPool(int max_threads)
        : max_threads_(std::max(1, max_threads)), scratch_(static_cast<size_t>(max_threads_)) {
        workers_.reserve(static_cast<size_t>(max_threads_ - 1));
        for (int id = 1; id < max_threads_; ++id)
            workers_.emplace_back([this, id] { worker_loop(id); });
    }

    ~MatmulPool() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }

    void run(const uint8_t* raw, const float* vectors, float* out, int rows,
             int cols, int tokens, int bytes_per_row, int threads, MatmulRowFn fn) {
        std::lock_guard<std::mutex> serial(run_mutex_);
        threads = std::max(1, std::min(std::min(threads, rows), max_threads_));
        if (rows < 256) threads = 1;
        if (threads == 1) {
            work_rows(raw, vectors, out, 0, rows, cols, tokens, bytes_per_row, fn, scratch_[0]);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            raw_ = raw; vectors_ = vectors; out_ = out; rows_ = rows;
            cols_ = cols; tokens_ = tokens; bytes_per_row_ = bytes_per_row;
            fn_ = fn; active_threads_ = threads; remaining_ = threads - 1;
            ++generation_;
        }
        start_cv_.notify_all();
        work_rows(raw, vectors, out, 0, rows / threads, cols, tokens, bytes_per_row, fn, scratch_[0]);
        std::unique_lock<std::mutex> lock(state_mutex_);
        done_cv_.wait(lock, [this] { return remaining_ == 0; });
    }

private:
    static void work_rows(const uint8_t* raw, const float* vectors, float* out,
                          int begin, int end, int cols, int tokens, int bytes_per_row,
                          MatmulRowFn fn, std::vector<float>& weights) {
        if (weights.size() < static_cast<size_t>(cols)) weights.resize(static_cast<size_t>(cols));
        for (int r = begin; r < end; ++r)
            fn(raw + static_cast<size_t>(r) * bytes_per_row, vectors,
               out + static_cast<size_t>(r) * tokens, cols, tokens, weights.data());
    }

    void worker_loop(int id) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            start_cv_.wait(lock, [this, &seen_generation] {
                return stopping_ || generation_ != seen_generation;
            });
            if (stopping_) return;
            seen_generation = generation_;
            if (id >= active_threads_) continue;
            const auto raw = raw_; const auto vectors = vectors_; auto out = out_;
            const int rows = rows_, cols = cols_, tokens = tokens_;
            const int bytes_per_row = bytes_per_row_, threads = active_threads_;
            const auto fn = fn_;
            lock.unlock();
            const int begin = static_cast<int>((static_cast<long long>(rows) * id) / threads);
            const int end = static_cast<int>((static_cast<long long>(rows) * (id + 1)) / threads);
            work_rows(raw, vectors, out, begin, end, cols, tokens, bytes_per_row, fn, scratch_[id]);
            lock.lock();
            if (--remaining_ == 0) done_cv_.notify_one();
        }
    }

    const int max_threads_;
    std::vector<std::thread> workers_;
    std::vector<std::vector<float>> scratch_;
    std::mutex run_mutex_, state_mutex_;
    std::condition_variable start_cv_, done_cv_;
    bool stopping_ = false;
    uint64_t generation_ = 0;
    int active_threads_ = 1, remaining_ = 0;
    const uint8_t* raw_ = nullptr;
    const float* vectors_ = nullptr;
    float* out_ = nullptr;
    int rows_ = 0, cols_ = 0, tokens_ = 0, bytes_per_row_ = 0;
    MatmulRowFn fn_ = nullptr;
};

static float dot_decoded_row(const float* weights, const float* x, int cols) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= cols; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(weights + i), _mm256_loadu_ps(x + i), acc);
    float sum = hsum256(acc);
    for (; i < cols; ++i) sum += weights[i] * x[i];
    return sum;
}

static void decode_q8_0_row(const uint8_t* row, float* weights, int cols) {
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t* p = row + b * 34;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
        for (int i = 0; i < 32; ++i) weights[b * 32 + i] = d * q[i];
    }
}
static void decode_q4_0_row(const uint8_t* row, float* weights, int cols) {
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t* p = row + b * 18;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const uint8_t* q = p + 2;
        for (int i = 0; i < 16; ++i) {
            weights[b * 32 + i] = d * (static_cast<int>(q[i] & 0x0f) - 8);
            weights[b * 32 + i + 16] = d * (static_cast<int>(q[i] >> 4) - 8);
        }
    }
}
static void decode_q4k_row(const uint8_t* row, float* weights, int cols) {
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t* p = row + b * 144;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const float dm = half_to_float(*reinterpret_cast<const uint16_t*>(p + 2));
        const uint8_t* scales = p + 4; const uint8_t* qs = p + 16;
        for (int g = 0; g < 8; ++g) {
            const float scale = d * q4_scale(scales, g), minimum = dm * q4_min(scales, g);
            const int qbase = (g / 2) * 32; const bool high = (g & 1) != 0;
            for (int j = 0; j < 32; ++j) {
                const uint8_t packed = qs[qbase + j];
                const int q = high ? (packed >> 4) : (packed & 0x0f);
                weights[b * 256 + g * 32 + j] = scale * q - minimum;
            }
        }
    }
}
static void decode_q5k_row(const uint8_t* row, float* weights, int cols) {
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t* p = row + b * 176;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p));
        const float dm = half_to_float(*reinterpret_cast<const uint16_t*>(p + 2));
        const uint8_t* scales = p + 4; const uint8_t* qh = p + 16; const uint8_t* qs = p + 48;
        for (int g = 0; g < 8; ++g) {
            const float scale = d * q4_scale(scales, g), minimum = dm * q4_min(scales, g);
            const int qbase = (g / 2) * 32; const bool high = (g & 1) != 0;
            for (int j = 0; j < 32; ++j) {
                const uint8_t packed = qs[qbase + j];
                const int low = high ? (packed >> 4) : (packed & 0x0f);
                const int q = low | (((qh[j] >> g) & 1) << 4);
                weights[b * 256 + g * 32 + j] = scale * q - minimum;
            }
        }
    }
}
static void decode_q6k_row(const uint8_t* row, float* weights, int cols) {
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t* p = row + b * 210;
        const uint8_t* ql = p; const uint8_t* qh = p + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(p + 192);
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(p + 208));
        for (int half = 0; half < 2; ++half) {
            for (int quad = 0; quad < 4; ++quad) {
                const int qlbase = half * 64 + ((quad == 1 || quad == 3) ? 32 : 0);
                const int qhbase = half * 32; const int scalebase = half * 8;
                for (int j = 0; j < 32; ++j) {
                    const uint8_t lo = ql[qlbase + j];
                    const int low = quad >= 2 ? (lo >> 4) : (lo & 0x0f);
                    const int high = (qh[qhbase + j] >> (quad * 2)) & 0x03;
                    const int q = (low | (high << 4)) - 32;
                    const int scale = scales[scalebase + quad * 2 + j / 16];
                    weights[b * 256 + half * 128 + quad * 32 + j] = d * scale * q;
                }
            }
        }
    }
}

template <typename DecodeFn>
static void decoded_matmul_row(const uint8_t* row, const float* vectors, float* out,
                               int cols, int tokens, float* weights, DecodeFn decode) {
    decode(row, weights, cols);
    // Reuse each decoded weight vector across eight prompt positions. The old
    // token-outer loop streamed the same 16-64 KiB row through the cache once
    // per token; this micro-kernel loads it once for eight independent dots.
    int t = 0;
    for (; t + 8 <= tokens; t += 8) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const float* x0 = vectors + static_cast<size_t>(t) * cols;
        const float* x1 = x0 + cols, *x2 = x1 + cols, *x3 = x2 + cols;
        const float* x4 = x3 + cols, *x5 = x4 + cols, *x6 = x5 + cols, *x7 = x6 + cols;
        int i = 0;
        for (; i + 8 <= cols; i += 8) {
            const __m256 w = _mm256_loadu_ps(weights + i);
            a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0 + i), a0);
            a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1 + i), a1);
            a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2 + i), a2);
            a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x3 + i), a3);
            a4 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x4 + i), a4);
            a5 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x5 + i), a5);
            a6 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x6 + i), a6);
            a7 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x7 + i), a7);
        }
        float sums[8] = {hsum256(a0), hsum256(a1), hsum256(a2), hsum256(a3),
                         hsum256(a4), hsum256(a5), hsum256(a6), hsum256(a7)};
        for (; i < cols; ++i) {
            sums[0] += weights[i] * x0[i]; sums[1] += weights[i] * x1[i];
            sums[2] += weights[i] * x2[i]; sums[3] += weights[i] * x3[i];
            sums[4] += weights[i] * x4[i]; sums[5] += weights[i] * x5[i];
            sums[6] += weights[i] * x6[i]; sums[7] += weights[i] * x7[i];
        }
        for (int k = 0; k < 8; ++k) out[t + k] = sums[k];
    }
    // A four-token tail keeps reuse high for prompt lengths not divisible by 8.
    for (; t + 4 <= tokens; t += 4) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const float* x0 = vectors + static_cast<size_t>(t) * cols;
        const float* x1 = x0 + cols, *x2 = x1 + cols, *x3 = x2 + cols;
        int i = 0;
        for (; i + 8 <= cols; i += 8) {
            const __m256 w = _mm256_loadu_ps(weights + i);
            a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0 + i), a0);
            a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1 + i), a1);
            a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2 + i), a2);
            a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x3 + i), a3);
        }
        float sums[4] = {hsum256(a0), hsum256(a1), hsum256(a2), hsum256(a3)};
        for (; i < cols; ++i) {
            sums[0] += weights[i] * x0[i]; sums[1] += weights[i] * x1[i];
            sums[2] += weights[i] * x2[i]; sums[3] += weights[i] * x3[i];
        }
        out[t] = sums[0]; out[t + 1] = sums[1]; out[t + 2] = sums[2]; out[t + 3] = sums[3];
    }
    for (; t < tokens; ++t)
        out[t] = dot_decoded_row(weights, vectors + static_cast<size_t>(t) * cols, cols);
}
static void q8_0_matmul_row(const uint8_t* row, const float* vectors, float* out, int cols, int tokens, float* weights) {
    decoded_matmul_row(row, vectors, out, cols, tokens, weights, decode_q8_0_row);
}
static void q4_0_matmul_row(const uint8_t* row, const float* vectors, float* out, int cols, int tokens, float* weights) {
    decoded_matmul_row(row, vectors, out, cols, tokens, weights, decode_q4_0_row);
}
static void q4k_matmul_row(const uint8_t* row, const float* vectors, float* out, int cols, int tokens, float* weights) {
    decoded_matmul_row(row, vectors, out, cols, tokens, weights, decode_q4k_row);
}
static void q5k_matmul_row(const uint8_t* row, const float* vectors, float* out, int cols, int tokens, float* weights) {
    decoded_matmul_row(row, vectors, out, cols, tokens, weights, decode_q5k_row);
}
static void q6k_matmul_row(const uint8_t* row, const float* vectors, float* out, int cols, int tokens, float* weights) {
    decoded_matmul_row(row, vectors, out, cols, tokens, weights, decode_q6k_row);
}

static void parallel_matmul(const uint8_t* raw, const float* vectors, float* out,
                            int rows, int cols, int tokens, int bytes_per_row,
                            int threads, MatmulRowFn fn) {
    static MatmulPool pool(static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency()))));
    pool.run(raw, vectors, out, rows, cols, tokens, bytes_per_row, threads, fn);
}

extern "C" __declspec(dllexport)
void localai_q8_0_matvec(const uint8_t* raw, const float* x, float* out,
                         int rows, int cols, int threads) {
    parallel_matvec(raw, x, out, rows, cols, (cols / 32) * 34, threads, q8_0_row);
}

extern "C" __declspec(dllexport)
void localai_q4_0_matvec(const uint8_t* raw, const float* x, float* out,
                         int rows, int cols, int threads) {
    const int blocks = cols / 32;
    auto augmented = quantize_input(x, cols, blocks, 32);
    parallel_matvec(raw, augmented.data(), out, rows, cols, blocks * 18, threads, q4_0_row);
}

extern "C" __declspec(dllexport)
void localai_q4k_matvec(const uint8_t* raw, const float* x, float* out,
                        int rows, int cols, int threads) {
    const int groups = (cols / 256) * 8;
    auto augmented = quantize_input_q8(x, cols, groups);
    parallel_matvec(raw, augmented.data(), out, rows, cols, (cols / 256) * 144, threads, q4k_row);
}
extern "C" __declspec(dllexport)
void localai_q5k_matvec(const uint8_t* raw, const float* x, float* out,
                        int rows, int cols, int threads) {
    const int groups = (cols / 256) * 8;
    auto augmented = quantize_input_q8(x, cols, groups);
    parallel_matvec(raw, augmented.data(), out, rows, cols, (cols / 256) * 176, threads, q5k_row);
}
extern "C" __declspec(dllexport)
void localai_q6k_matvec(const uint8_t* raw, const float* x, float* out,
                        int rows, int cols, int threads) {
    auto augmented = quantize_input(x, cols, 0, 16);
    parallel_matvec(raw, augmented.data(), out, rows, cols, (cols / 256) * 210, threads, q6k_row);
}





extern "C" __declspec(dllexport)
void localai_q8_0_matmul(const uint8_t* raw, const float* vectors, float* out,
                         int rows, int cols, int tokens, int threads) {
    parallel_matmul(raw, vectors, out, rows, cols, tokens, (cols / 32) * 34, threads, q8_0_matmul_row);
}
extern "C" __declspec(dllexport)
void localai_q4_0_matmul(const uint8_t* raw, const float* vectors, float* out,
                         int rows, int cols, int tokens, int threads) {
    parallel_matmul(raw, vectors, out, rows, cols, tokens, (cols / 32) * 18, threads, q4_0_matmul_row);
}
extern "C" __declspec(dllexport)
void localai_q4k_matmul(const uint8_t* raw, const float* vectors, float* out,
                        int rows, int cols, int tokens, int threads) {
    parallel_matmul(raw, vectors, out, rows, cols, tokens, (cols / 256) * 144, threads, q4k_matmul_row);
}
extern "C" __declspec(dllexport)
void localai_q5k_matmul(const uint8_t* raw, const float* vectors, float* out,
                        int rows, int cols, int tokens, int threads) {
    parallel_matmul(raw, vectors, out, rows, cols, tokens, (cols / 256) * 176, threads, q5k_matmul_row);
}
extern "C" __declspec(dllexport)
void localai_q6k_matmul(const uint8_t* raw, const float* vectors, float* out,
                        int rows, int cols, int tokens, int threads) {
    parallel_matmul(raw, vectors, out, rows, cols, tokens, (cols / 256) * 210, threads, q6k_matmul_row);
}

