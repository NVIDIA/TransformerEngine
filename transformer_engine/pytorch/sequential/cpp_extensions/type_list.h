#include <cstddef>
#include <type_traits>

template <typename... Ts> struct type_list;

template <typename TL> struct type_list_front;
template <typename TL> struct type_list_back;
template <typename TL> struct type_list_reverse_list;
template <typename TL, size_t I> struct type_list_index;
template <typename TL1, typename TL2> struct type_list_cat_list;
template <typename TL, size_t N = 1> struct type_list_pop_front_list;
template <typename TL, size_t N = 1> struct type_list_pop_back_list;

template <typename First, typename... Ts>
struct type_list_front<type_list<First, Ts...>> {
  using type = First;
};

template <typename First, typename... Ts>
struct type_list_pop_front_list<type_list<First, Ts...>, 0> {
  using type = type_list<First, Ts...>;
};

template <typename First, typename... Ts, size_t N>
struct type_list_pop_front_list<type_list<First, Ts...>, N> {
  using type = typename type_list_pop_front_list<type_list<Ts...>, N - 1>::type;
};

template <typename... Ts, size_t I>
struct type_list_index<type_list<Ts...>, I> {
private:
  using stripped = typename type_list_pop_front_list<type_list<Ts...>, I>::type;

public:
  using type = typename type_list_front<stripped>::type;
};

template <typename... Ts1, typename... Ts2>
struct type_list_cat_list<type_list<Ts1...>, type_list<Ts2...>> {
  using type = type_list<Ts1..., Ts2...>;
};

template <typename First, typename... Ts>
struct type_list_reverse_list<type_list<First, Ts...>> {
private:
  using ts_reversed = typename type_list_reverse_list<type_list<Ts...>>::type;
  using back_list = type_list<First>;

public:
  using type = typename type_list_cat_list<ts_reversed, back_list>::type;
};
template <> struct type_list_reverse_list<type_list<>> {
  using type = type_list<>;
};

template <typename... Ts> struct type_list_back<type_list<Ts...>> {
private:
  using reversed = typename type_list_reverse_list<type_list<Ts...>>::type;

public:
  using type = typename type_list_front<reversed>::type;
};

template <typename... Ts, size_t N>
struct type_list_pop_back_list<type_list<Ts...>, N> {
private:
  using reversed = typename type_list_reverse_list<type_list<Ts...>>::type;
  using stripped = typename type_list_pop_front_list<reversed, N>::type;

public:
  using type = typename type_list_reverse_list<stripped>::type;
};

template <typename TL>
using type_list_front_t = typename type_list_front<TL>::type;
template <typename TL>
using type_list_back_t = typename type_list_back<TL>::type;
template <typename TL>
using type_list_reverse_list_t = typename type_list_reverse_list<TL>::type;
template <typename TL, size_t I>
using type_list_index_t = typename type_list_index<TL, I>::type;
template <typename TL1, typename TL2>
using type_list_cat_list_t = typename type_list_cat_list<TL1, TL2>::type;
template <typename TL, size_t N = 1>
using type_list_pop_front_list_t =
    typename type_list_pop_front_list<TL, N>::type;
template <typename TL, size_t N = 1>
using type_list_pop_back_list_t = typename type_list_pop_back_list<TL, N>::type;

template <typename... Ts> struct type_list {
  using front = type_list<type_list_front_t<type_list>>;
  using front_t = type_list_index_t<front, 0>;

  using back = type_list<type_list_back_t<type_list>>;
  using back_t = type_list_index_t<back, 0>;

  template <size_t N = 1>
  using pop_front = type_list_pop_front_list_t<type_list, N>;

  template <size_t N = 1>
  using pop_back = type_list_pop_back_list_t<type_list, N>;
};
