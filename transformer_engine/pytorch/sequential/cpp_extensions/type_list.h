#include <cstddef>
#include <initializer_list>
#include <type_traits>

template <typename... Ts> struct type_list;

template <typename TL> struct type_list_front;
template <typename TL> struct type_list_back;
template <typename TL> struct type_list_reverse_list;
template <typename TL, size_t I> struct type_list_index;
template <typename TL1, typename TL2> struct type_list_cat_list;
template <typename TL, size_t N = 1> struct type_list_pop_front_list;
template <typename TL, size_t N = 1> struct type_list_pop_back_list;
template <typename TL, typename T> struct type_list_contains;
template <typename TL, template <typename> typename Pred> struct type_list_any;
template <typename TL, typename T> struct type_list_find;
template <typename TL, template <typename> typename Pred>
struct type_list_first;

template <typename First, typename... Ts>
struct type_list_front<type_list<First, Ts...>> {
  using type = First;
};

template <typename First, typename... Ts>
struct type_list_pop_front_list<type_list<First, Ts...>, 0> {
  using type = type_list<First, Ts...>;
};
template <> struct type_list_pop_front_list<type_list<>, 0> {
  using type = type_list<>;
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

template <typename... Ts, template <typename> typename Pred>
struct type_list_any<type_list<Ts...>, Pred> {
  static constexpr bool value = (Pred<Ts>::value || ...);
};

template <typename... Ts, template <typename> typename Pred>
struct type_list_first<type_list<Ts...>, Pred> {
private:
  static constexpr bool values[] = {Pred<Ts>::value...};

public:
  static constexpr size_t value = []() {
    for (size_t i = 0; i < sizeof(values) / sizeof(bool); ++i) {
      if (values[i]) {
        return i;
      }
    }
    return sizeof(values) / sizeof(bool);
  }();
};

template <typename... Ts, typename T>
struct type_list_contains<type_list<Ts...>, T> {
private:
  template <typename U> struct pred {
    static constexpr bool value = std::is_same_v<T, U>;
  };

public:
  static constexpr bool value = type_list_any<type_list<Ts...>, pred>::value;
};

template <typename... Ts, typename T>
struct type_list_find<type_list<Ts...>, T> {
  template <typename U> struct pred {
    static constexpr bool value = std::is_same_v<T, U>;
  };

public:
  static constexpr size_t value =
      type_list_first<type_list<Ts...>, pred>::value;
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
template <typename TL, typename T>
constexpr bool type_list_contains_v = type_list_contains<TL, T>::value;
template <typename TL, template <typename> typename Pred>
constexpr bool type_list_any_v = type_list_any<TL, Pred>::value;
template <typename TL, typename T>
constexpr size_t type_list_find_v = type_list_find<TL, T>::value;
template <typename TL, template <typename> typename Pred>
constexpr size_t type_list_first_v = type_list_first<TL, Pred>::value;

template <typename... Ts> struct type_list {
  using front = type_list<type_list_front_t<type_list>>;
  using front_t = type_list_index_t<front, 0>;

  using back = type_list<type_list_back_t<type_list>>;
  using back_t = type_list_index_t<back, 0>;

  using reverse = type_list_reverse_list_t<type_list>;

  template <size_t I> using get = type_list_index_t<type_list, I>;

  template <size_t N = 1>
  using pop_front = type_list_pop_front_list_t<type_list, N>;

  template <size_t N = 1>
  using pop_back = type_list_pop_back_list_t<type_list, N>;

  template <typename T>
  static constexpr bool contains = type_list_contains_v<type_list, T>;

  template <template <typename> typename Pred>
  static constexpr bool any = type_list_any_v<type_list, Pred>;

  template <typename T>
  static constexpr size_t find = type_list_find_v<type_list, T>;

  template <typename T, template <typename> typename Pred>
  static constexpr size_t first = type_list_first_v<type_list, Pred>;

  static constexpr size_t size = sizeof...(Ts);
};
template <> struct type_list<> {};
