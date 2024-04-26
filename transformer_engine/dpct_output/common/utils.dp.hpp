/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTILS_CUH_
#define TRANSFORMER_ENGINE_COMMON_UTILS_CUH_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#if !defined(__CUDACC_RTC__)
#include <cstdint>
#else
// Importing C++ standard headers is a pain with NVRTC
using uint8_t = unsigned char;
using uint16_t = unsigned short int;  // NOLINT(*)
using uint32_t = unsigned int;
using uint64_t = unsigned long long int;  // NOLINT(*)
static_assert(sizeof(uint8_t) == 1);
static_assert(sizeof(uint16_t) == 2);
static_assert(sizeof(uint32_t) == 4);
static_assert(sizeof(uint64_t) == 8);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t THREADS_PER_WARP = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1011:219: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(const sycl::float2 &a,
                              const sycl::float2 &b) { // NOLINT(*)
    return {a.x() + b.x(), a.y() + b.y()};
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1011:220: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float2 &a, const sycl::float2 &b) { // NOLINT(*)
    a.x() += b.x();
    a.y() += b.y();
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Sum {
    inline Sum() {}
    inline T operator()(const T &a, const T &b) const {
        return dpct_operator_overloading::operator+(a, b);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline T warp_shuffle_xor(const T & x, uint32_t idx,
                          const sycl::nd_item<3> &item_ct1) {
    return dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, idx);
}

template <>
inline sycl::float2
warp_shuffle_xor<sycl::float2>(const sycl::float2 &x, uint32_t idx,
                               const sycl::nd_item<3> &item_ct1) {
    return {warp_shuffle_xor(x.x(), idx, item_ct1),
            warp_shuffle_xor(x.y(), idx, item_ct1)};
}

template<typename T>
inline T warp_shuffle_down(const T & x, uint32_t idx,
                           const sycl::nd_item<3> &item_ct1) {
    /*
    DPCT1121:0: Make sure that the "x" which is used in the SYCL group
    function/algorithm is initialized.
    */
    return dpct::shift_sub_group_left(item_ct1.get_sub_group(), x, idx);
}

template <>
inline sycl::float2
warp_shuffle_down<sycl::float2>(const sycl::float2 &x, uint32_t idx,
                                const sycl::nd_item<3> &item_ct1) {
    return {warp_shuffle_down(x.x(), idx, item_ct1),
            warp_shuffle_down(x.y(), idx, item_ct1)};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace transformer_engine {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint16 {
    sycl::uint4 u;
    sycl::uint4 v;
    sycl::uint4 s;
    sycl::uint4 t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint8 {
    sycl::uint4 u;
    sycl::uint4 v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES>
struct BytesToType {};

template<>
struct BytesToType<64> {
    using Type = uint16;
    static_assert(sizeof(Type) == 64);
};

template<>
struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
};

template<>
struct BytesToType<16> {
    using Type = sycl::uint4;
    static_assert(sizeof(Type) == 16);
};

template<>
struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<>
struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<>
struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<>
struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct TypeToVec2 {};

template<>
struct TypeToVec2<float> {
    using Type = sycl::float2;
};

template <> struct TypeToVec2<sycl::half> {
    using Type = sycl::half2;
};

template<>
struct TypeToVec2<nv_bfloat16> {
    using Type = nv_bfloat162;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int INDEX>
struct Get {
    template<typename T, typename R>
    static inline R of(const T &vec);
};

template<>
template<typename T, typename R>
inline R Get<0>::of(const T &vec) {
    return vec.x();
}

template<>
template<typename T, typename R>
inline R Get<1>::of(const T &vec) {
    return vec.y();
}

template<>
template<typename T, typename R>
inline R Get<2>::of(const T &vec) {
    return vec.z;
}

template<>
template<typename T, typename R>
inline R Get<3>::of(const T &vec) {
    return vec.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Src, typename Dst>
struct Converter{
    static inline Dst convert(const Src &from) {
        return Dst(from);
    }
};

template <> struct Converter<sycl::float2, sycl::half2> {
    static inline sycl::half2 convert(const sycl::float2 &x) {
        return x.convert<sycl::half, sycl::rounding_mode::rte>();
    }
};

template <> struct Converter<sycl::float2, nv_bfloat162> {
    static inline nv_bfloat162 convert(const sycl::float2 &x) {
#if DPCT_COMPATIBILITY_TEMP >= 800
        return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(x[0], x[1]);
#else
        union {
            nv_bfloat162 raw;
            nv_bfloat16 elt[2];
        } tmp;
        tmp.elt[0] = __float2bfloat16_rn(x.x);
        tmp.elt[1] = __float2bfloat16_rn(x.y);
        return tmp.raw;
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Zeros{
    static inline T get() {
        return T(0.f);
    }
};

template <> struct Zeros<sycl::float2> {
    static inline sycl::float2 get() {
        return sycl::float2(0.f, 0.f);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Elt_type, uint32_t NUM_ELT>
struct Vec {
    enum { BYTES = NUM_ELT * sizeof(Elt_type) };

    using Vec_type = typename BytesToType<BYTES>::Type;
    using type = Elt_type;

    using Alias_type = union {
        Vec_type vec;
        Elt_type elt[NUM_ELT];
    };

    Alias_type data;

    template<typename S>
    inline void to(Vec<S, NUM_ELT> &other) {  // NOLINT(*)
        #pragma unroll
        for ( int it = 0; it < NUM_ELT; it++ ) {
            other.data.elt[it] = S(this->data.elt[it]);
        }
    }

    template<typename Op>
    inline void assign(const Op &op) {
        #pragma unroll
        for ( int it = 0; it < NUM_ELT; it++ ) {
            this->data.elt[it] = op(it);
        }
    }

    // Pointer is cast to vector type
    inline void load_from(const void *base_ptr, size_t idx = 0) {
        this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
    }

    // Pointer is cast to vector type
    inline void store_to(void *base_ptr, size_t idx = 0) const {
        static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
    }

    // Pointer is cast to element type. Loads min(count, NUM_ELT)
    // elements and any remaining elements are set to zero.
    inline void load_from_elts(const void *base_ptr,
                                          size_t idx = 0,
                                          size_t count = NUM_ELT) {
        const Elt_type *elt_ptr = static_cast<const Elt_type *>(base_ptr) + idx;
        if ( count < NUM_ELT
             || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0 ) {
            #pragma unroll
            for ( int it = 0; it < NUM_ELT; it++ ) {
                this->data.elt[it] = (it < count
                                      ? elt_ptr[it]
                                      : Elt_type(0.f));
            }
        } else {
            this->load_from(elt_ptr);
        }
    }

    // Pointer is cast to element type. Stores min(count, NUM_ELT)
    // elements.
    inline void store_to_elts(void *base_ptr,
                                         size_t idx = 0,
                                         size_t count = NUM_ELT) const {
        Elt_type *elt_ptr = static_cast<Elt_type *>(base_ptr) + idx;
        if ( count < NUM_ELT
             || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0 ) {
            #pragma unroll
            for ( int it = 0; it < NUM_ELT; it++ ) {
                if ( it < count ) {
                    elt_ptr[it] = this->data.elt[it];
                }
            }
        } else {
            this->store_to(elt_ptr);
        }
    }

    inline void clear() {
        #pragma unroll
        for ( int it = 0; it < NUM_ELT; it++ ) {
            this->data.elt[it] = Elt_type(0.f);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct InterCTASync {
    inline InterCTASync(int *barrier,
                                   int group,
                                   int num_groups,
                                   int group_size)
        : phase_counter_(0)
        , b0_(barrier + group)  // The barrier for this group of CTAs.
        , b1_(barrier + group + num_groups)  // The barrier for this group of CTAs.
        , group_size_(group_size) {
        // BARRIERS ARE ASSUMED TO BE INITIALIZED TO 0!
    }

    inline void spin_wait_(int *barrier, int step, int expected) {
        /*
        DPCT1053:1: Migration of device assembly code is not supported.
        */
        asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(barrier),
                     "r"(step));
        for ( int found = -1; found != expected; ) {
            /*
            DPCT1053:2: Migration of device assembly code is not supported.
            */
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];"
                         : "=r"(found)
                         : "l"(barrier));
        }
    }

    inline void sync(const sycl::nd_item<3> &item_ct1) {
        // ALL THREADS MUST ENTER!

        // We switch barrier every iteration.
        int *barrier = phase_counter_ & 0x1 ? b1_ : b0_;
        // We decrement every other iteration.
        bool dec = phase_counter_ & 0x2;
        int step = dec ? -1 : 1;
        int expected = dec ? 0 : group_size_;
        // There are only 4 phases: up/down for b0/b1.
        phase_counter_ = (phase_counter_ + 1) & 0x3;

        if (item_ct1.get_local_id(2) == 0) {
            spin_wait_(barrier, step, expected);
        }
        // CTA waits for thread 0
        /*
        DPCT1065:221: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    int phase_counter_;
    int * b0_;
    int * b1_;
    int group_size_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t CTAS_PER_ROW, uint32_t WARPS_M, uint32_t WARPS_N>
struct Reducer : public Reducer<T, 1, WARPS_M, WARPS_N> {
    using Base = Reducer<T, 1, WARPS_M, WARPS_N>;
    using Type = typename Base::Type;

    enum { SMEM_BYTES = Base::SMEM_BYTES };

    enum { WS_BARRIER_BYTES = 2 * sizeof(int) };
    enum { WS_DATA_BYTES = WARPS_M * CTAS_PER_ROW * sizeof(T) };

    // size of the barriers + temporary result per CTA (multiply with CTAS_PER_ROW to get total)
    enum { WORKSPACE_BYTES_PER_GROUP = Base::WORKSPACE_BYTES_PER_GROUP + WS_BARRIER_BYTES +
                                       WS_DATA_BYTES };

    template<typename Params>
    inline Reducer(const Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                              uint32_t warp_n, uint32_t lane, void * smem)
        : Base(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , inter_cta_(params.barrier, bidm, params.ctas_per_col, CTAS_PER_ROW)
        , bidn_(bidn)  // CTA id within the group.
        , w0_(static_cast<T*>(params.workspace) + (bidm * WARPS_M + warp_m) * CTAS_PER_ROW)
        , w1_(w0_ + params.ctas_per_col * WARPS_M * CTAS_PER_ROW) {}

    template<typename Op>
    inline T allreduce(T data, const Op &op, const sycl::nd_item<3> &item_ct1) {
        data = Base::reduce(data, op, item_ct1);
        // We switch workspace every iteration.
        T * const workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

        // Warp leaders 0 hold the CTA-local results.
        if ( this->warp_n_ == 0 && this->lane_ == 0 ) {
            workspace[bidn_] = data;
        }
        inter_cta_.sync(item_ct1);
        static_assert(CTAS_PER_ROW <= 32);
        T total = Zeros<T>::get();
        if (this->lane_ < CTAS_PER_ROW) {
            total = workspace[this->lane_];
        }
        total = Reducer<T, 1, 1, 1>::allreduce_(total, op, item_ct1);

        return total;
    }

    InterCTASync inter_cta_;

    T * const w0_;
    T * const w1_;
    int bidn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M>
struct Reducer<T, 1, WARPS_M, 1> {
    using Type = T;
    enum { SMEM_BYTES = 0 };
    enum { WORKSPACE_BYTES_PER_GROUP = 0 };

    enum { THREADS_PER_WARP = 32 };

    template<typename Params>
    inline Reducer(const Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                              uint32_t warp_n, uint32_t lane, void * smem)
        : warp_n_(warp_n)
        , lane_(lane) {}

    template<typename Op>
    static inline T allreduce_(T data, const Op &op,
                               const sycl::nd_item<3> &item_ct1) {
        #pragma unroll
        for ( int it = 1; it < THREADS_PER_WARP; it *= 2 ) {
            data = op(data, warp_shuffle_xor(data, it, item_ct1));
        }
        return data;
    }

    template<typename Op>
    inline T allreduce(T data, const Op &op, const sycl::nd_item<3> &item_ct1) {
        return allreduce_(data, op, item_ct1);
    }

    template<typename Op>
    inline T reduce(T data, const Op &op, const sycl::nd_item<3> &item_ct1) {
        // only lane 0 holds the result!
        #pragma unroll
        for ( int it = THREADS_PER_WARP / 2; it > 0; it /= 2 ) {
            data = op(data, warp_shuffle_down(data, it, item_ct1));
        }
        return data;
    }
    int warp_n_;
    int lane_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct Reducer<T, 1, WARPS_M, WARPS_N> : public Reducer<T, 1, WARPS_M, 1> {
    using Base = Reducer<T, 1, WARPS_M, 1>;

    using Type = T;

    enum { SMEM_BYTES = Base::SMEM_BYTES + WARPS_M * WARPS_N * sizeof(T) * 2 };
    enum { WORKSPACE_BYTES_PER_GROUP = 0 };

    enum { THREADS_PER_WARP = 32 };

    template<typename Params>
    inline Reducer(const Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                              uint32_t warp_n, uint32_t lane, void * smem)
        : Base(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , use0_(true), smem0_(&(static_cast<T *>(smem)[warp_m * WARPS_N]))
        , smem1_(smem0_ + WARPS_M * WARPS_N) {}

    template<typename Op>
    inline T allreduce(T data, const Op & op, const sycl::nd_item<3> &item_ct1) {
        T * const smem = use0_ ? smem0_ : smem1_;
        use0_ = !use0_;
        data = Base::reduce(data, op, item_ct1);
        if ( this->lane_ == 0 ) {
            smem[this->warp_n_] = data;
        }
        /*
        DPCT1065:222: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        T out = Zeros<T>::get();
        #pragma unroll
        for ( int it = 0; it < WARPS_N; it++ ) {
            out = op(out, smem[it]);
        }
        return out;
    }

    template<typename Op>
    inline T reduce(T data, const Op &op, const sycl::nd_item<3> &item_ct1) {
        T * const smem = use0_ ? smem0_ : smem1_;
        use0_ = !use0_;
        // only intra-CTA group leader holds the result!
        data = Base::reduce(data, op, item_ct1);
        if ( this->lane_ == 0 ) {
            smem[this->warp_n_] = data;
        }
        /*
        DPCT1065:223: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        T out = Zeros<T>::get();
        if ( this->warp_n_ == 0 && this->lane_ == 0 ) {
            #pragma unroll
            for ( int it = 0; it < WARPS_N; it++ ) {
                out = op(out, smem[it]);
            }
        }
        return out;
    }

    T * const smem0_;
    T * const smem1_;
    bool use0_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct DynamicReducer : public Reducer<T, 1, WARPS_M, WARPS_N> {
    using Base = Reducer<T, 1, WARPS_M, WARPS_N>;
    using Type = typename Base::Type;

    template<typename Params>
    inline DynamicReducer(const Params & params,
                                     uint32_t bidm, uint32_t bidn,
                                     uint32_t warp_m, uint32_t warp_n,
                                     uint32_t lane, void * smem)
        : Base(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , inter_cta_(params.barrier, bidm, params.ctas_per_col, params.ctas_per_row)
        , bidn_(bidn)  // CTA id within the group.
        , w0_(static_cast<T*>(params.workspace) + (bidm * WARPS_M + warp_m) * params.ctas_per_row)
        , w1_(w0_ + params.ctas_per_col * WARPS_M * params.ctas_per_row) {}

    template<typename Op>
    inline T allreduce(T data, const Op &op, const sycl::nd_item<3> &item_ct1) {
        // Trivial case
        if (inter_cta_.group_size_ == 1) {
            /*
            DPCT1084:3: The function call "Reducer::allreduce" has multiple
            migration results in different template instantiations that could
            not be unified. You may need to adjust the code.
            */
            return Base::allreduce(data, op, item_ct1);
        }

        /*
        DPCT1084:4: The function call "Reducer::reduce" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        data = Base::reduce(data, op, item_ct1);
        // We switch workspace every iteration.
        T * const workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

        // Warp leaders 0 hold the CTA-local results.
        if ( this->warp_n_ == 0 && this->lane_ == 0 ) {
            workspace[bidn_] = data;
        }
        inter_cta_.sync(item_ct1);
        T total = Zeros<T>::get();
        for ( int it = this->lane_;
              it < inter_cta_.group_size_;
              it += THREADS_PER_WARP ) {
            total = op(total, workspace[it]);
        }
        total = Reducer<T, 1, 1, 1>::allreduce_(total, op, item_ct1);

        return total;
    }

    template<typename Op>
    inline T reduce(T data, const Op &op) {
        return allreduce(data, op);
    }

    InterCTASync inter_cta_;

    T * const w0_;
    T * const w1_;
    int bidn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
This is an implementation of the parallel Welford algorithm for incrementally computing variance

This algorithm is known as Chan's update formulae (Chat et al '79):
http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

An introduction is provided by Wikipedia here:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance?section=5#Parallel_algorithm

A detailed reference on the exact version implemented (with better numerical stability) is provided here:
https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
*/

template<typename T>
inline void warp_chan_upd_dynamic(T &m_a, T &m2_a, T &n_a, int num_active,
                                  const sycl::nd_item<3> &item_ct1) { // NOLINT(*)
    // Assume at least leftmost is valid and
    // init: step = next_pow2(num_active) / 2 (might get NaN otherwise)
    int highest_bit_set = (8 * sizeof(num_active)) - sycl::clz(num_active - 1);

#pragma unroll
    for ( int step = (1 << (highest_bit_set - 1)); step > 0; step /= 2 ) {
        // Exchange
        T n_b = warp_shuffle_down(n_a, step, item_ct1);
        T m_b = warp_shuffle_down(m_a, step, item_ct1);
        T m2_b = warp_shuffle_down(m2_a, step, item_ct1);

        // Update
        const T n_ab = n_a + n_b;  // We can handle one of them being 0, not both.
        // Might have different n per thread, otherwise this would simplify :(
        const T rn_ab = 1.f / n_ab;
        const T delta = m_a - m_b;
        const float m2_ab = m2_a + m2_b + delta * delta * n_a * n_b * rn_ab;
        const float m_ab = (n_a * m_a + n_b * m_b) * rn_ab;

        n_a = n_ab;
        m_a = m_ab;
        m2_a = m2_ab;
    }
    // Intra-warp broadcast (only lane 0 has valid stats).
    /*
    DPCT1121:5: Make sure that the "m_a" which is used in the SYCL group
    function/algorithm is initialized.
    */
    m_a = dpct::select_from_sub_group(item_ct1.get_sub_group(), m_a, 0);
    /*
    DPCT1121:6: Make sure that the "m2_a" which is used in the SYCL group
    function/algorithm is initialized.
    */
    m2_a = dpct::select_from_sub_group(item_ct1.get_sub_group(), m2_a, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t CTAS_PER_ROW, uint32_t WARPS_M, uint32_t WARPS_N>
struct Stats {
    // This could be done generically with the Reducer. But then we
    // would have to exchange 3 instead of 2 fields.

    using BlockStats = Stats<T, 1, WARPS_M, WARPS_N>;
    using stats_t = typename BlockStats::stats_t;

    enum { SMEM_BYTES = BlockStats::SMEM_BYTES };

    template<typename Params>
    inline Stats(const Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                            uint32_t warp_n, uint32_t lane, void * smem)
        : inter_cta_(params.barrier, bidm, params.ctas_per_col, CTAS_PER_ROW)
        , block_stats_(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , bidn_(bidn)  // CTA id within the group.
        , w0_(static_cast<stats_t*>(params.workspace) + (bidm * WARPS_M + warp_m) * CTAS_PER_ROW)
        , w1_(w0_ + params.ctas_per_col * WARPS_M * CTAS_PER_ROW)
        , warp_n_(warp_n)
        , lane_(lane) {}

    template<uint32_t N>
    inline stats_t compute(const T (&elts)[N], const T rn,
                           const sycl::nd_item<3> &item_ct1) {
        constexpr T ELTS_PER_ROW_PER_CTA = N * WARPS_N * THREADS_PER_WARP;
        // TODO(ptredak) rn is not really needed here..
        constexpr T block_rn = 1.f / T(ELTS_PER_ROW_PER_CTA);
        stats_t block_stats = block_stats_.compute(elts, block_rn, item_ct1);

        stats_t * const workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

        if ( warp_n_ == 0 && lane_ == 0 ) {
            workspace[bidn_] = block_stats;
        }

        // Wait for all CTAS_PER_ROW CTAS in the group to have written their result.
        inter_cta_.sync(item_ct1);

        T n = Zeros<T>::get();
        T m = Zeros<T>::get();
        T m2 = Zeros<T>::get();

        // Assume CTA group size in N less than 32, such that we can finalize with a single warp.
        static_assert(CTAS_PER_ROW <= 32);

        // Every warp does the final reduction locally.
        if ( lane_ < CTAS_PER_ROW ) {
            stats_t result = workspace[lane_];
            n = ELTS_PER_ROW_PER_CTA;
            m = transformer_engine::Get<0>::of<stats_t, T>(result);
            m2 = transformer_engine::Get<1>::of<stats_t, T>(result);
        }

        warp_chan_upd_dynamic(m, m2, n, CTAS_PER_ROW, item_ct1);

        return { m, m2 };
    }

    InterCTASync inter_cta_;
    BlockStats block_stats_;

    stats_t * const w0_;
    stats_t * const w1_;
    int bidn_;
    int warp_n_;
    int lane_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct Stats<T, 1, WARPS_M, WARPS_N> {
    using WarpStats = Stats<T, 1, WARPS_M, 1>;
    using stats_t = typename WarpStats::stats_t;

    enum { SMEM_BYTES = WARPS_M * WARPS_N * sizeof(stats_t) * 2 };

    template<typename Params>
    inline Stats(const Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                            uint32_t warp_n, uint32_t lane, void * smem)
        : warp_stats_(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , use0_(true) {
        smem0_ = static_cast<stats_t*>(smem) + warp_m * WARPS_N;
        smem1_ = smem0_ + WARPS_M * WARPS_N;
    }

    template<uint32_t N>
    inline stats_t compute(const T (&elts)[N], const T rn,
                           const sycl::nd_item<3> &item_ct1) {
        stats_t * smem = use0_ ? smem0_ : smem1_;
        use0_ = !use0_;
        // Compute warp local for all WARPS_N
        constexpr T warp_rn = 1.f / T(N * THREADS_PER_WARP);
        stats_t warp_stats = warp_stats_.compute(elts, warp_rn, item_ct1);

        // Each warp warp leader stores its stats
        const auto warp_n = warp_stats_.reducer_.warp_n_;
        const auto lane = warp_stats_.reducer_.lane_;
        if ( lane == 0 ) {
            smem[warp_n] = warp_stats;
        }
        /*
        DPCT1065:224: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        T n = Zeros<T>::get();
        T m = Zeros<T>::get();
        T m2 = Zeros<T>::get();

        // Assume that there are less than 32 warps, such that we can finalize with a single warp
        static_assert(WARPS_N <= 32);
        if (lane < WARPS_N) {
            stats_t result = smem[lane];
            n = N * THREADS_PER_WARP;
            m = transformer_engine::Get<0>::of<stats_t, T>(result);
            m2 = transformer_engine::Get<1>::of<stats_t, T>(result);
        }

        warp_chan_upd_dynamic(m, m2, n, WARPS_N, item_ct1);

        return { m, m2 };
    }
    WarpStats warp_stats_;
    stats_t * smem0_;
    stats_t * smem1_;
    bool use0_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M>
struct Stats<T, 1, WARPS_M, 1> {
    using stats_t = typename TypeToVec2<T>::Type;
    // The simple Warp reducer.
    using Reducer = Reducer<T, 1, WARPS_M, 1>;

    enum { SMEM_BYTES = 0 };

    template<typename Params>
    inline Stats(const Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                            uint32_t warp_n, uint32_t lane, void * smem)
        : reducer_(params, bidm, bidn, warp_m, warp_n, lane, smem) {}

    template<uint32_t N>
    inline stats_t compute(const T (&elts)[N], const T rn,
                           const sycl::nd_item<3> &item_ct1) {
        auto sum = Sum<T>();

        T m = Zeros<T>::get();
        #pragma unroll
        for ( int it = 0; it < N; it++ ) {
            m += elts[it];
        }
        m = reducer_.allreduce(m, sum, item_ct1) * rn;

        T m2 = Zeros<T>::get();
        #pragma unroll
        for ( int it = 0; it < N; it++ ) {
            T diff = (elts[it] - m);
            m2 += diff * diff;
        }
        m2 = reducer_.allreduce(m2, sum, item_ct1);

        return {m, m2};
    }

    Reducer reducer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int num_elems>
__dpct_inline__ float warp_reduce_max(const float m,
                                      const sycl::nd_item<3> &item_ct1) {
    float tmp = m;
#pragma unroll
    for (int delta = num_elems/2; delta > 0; delta /= 2) {
        const float other_m =
            dpct::shift_sub_group_left(item_ct1.get_sub_group(), tmp, delta);
        __builtin_assume(tmp >= 0);
        __builtin_assume(other_m >= 0);
        tmp = sycl::fmax(tmp, (float)other_m);
    }
    return tmp;
}

template <int num_warps, typename compute_t>
__dpct_inline__ compute_t reduce_max(const compute_t m, const int warpid,
                                     const sycl::nd_item<3> &item_ct1,
                                     float *staging) {

    constexpr int warp_size = 32;
    const float my_max = m;
    const float my_warp_max = warp_reduce_max<warp_size>(my_max, item_ct1);
    if (item_ct1.get_local_id(2) % 32 == 0) {
        staging[warpid] = my_warp_max;
    }
    /*
    DPCT1065:225: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    compute_t result = 0;
    if (warpid == 0) {
        const float my_max = item_ct1.get_local_id(2) < num_warps
                                 ? staging[item_ct1.get_local_id(2)]
                                 : 0;
        result = warp_reduce_max<num_warps>(my_max, item_ct1);
    }
    return result;
}

// Works only on positive values
__dpct_inline__ void atomicMaxFloat(float *addr, const float value) {
    dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
        reinterpret_cast<int *>(addr), sycl::bit_cast<int>(value));
}

// Works only on positive values
__dpct_inline__ void atomicMinFloat(float *addr, const float value) {
    dpct::atomic_fetch_min<sycl::access::address_space::generic_space>(
        reinterpret_cast<int *>(addr), sycl::bit_cast<int>(value));
}

template <typename T>
__dpct_inline__ void reciprocal(T *value_inv, const T value) {
    *value_inv = 1 / value;
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTILS_CUH_
