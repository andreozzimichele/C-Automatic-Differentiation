#pragma once

#include <iostream>

namespace autodiff {

template <typename T> class forwardNums {
private:
  T _value{};
  T _dv_dx{};

public:
  forwardNums<T>(T input_v = T{}, T input_dv = T{})
      : _value{input_v}, _dv_dx{input_dv} {}

  T get_v() const noexcept { return _value; }
  T get_dv() const noexcept { return _dv_dx; }
  void set_v(T const &input_v) noexcept { _value = input_v; }
  void set_dv(T const &input_dv) noexcept { _dv_dx = input_dv; }
  void print() const {
    std::cout << "Primal: " << _value << ", Dual: " << _dv_dx << "\n";
  }

  // Unary minus operator, as in a = - b;
  forwardNums<T> operator-() const { return forwardNums<T>(-_value, -_dv_dx); }

  // Overloaded operators are non-member friend functions, so that 5.0 * a is
  // possible
  friend forwardNums<T> operator+(const forwardNums<T> &lhs,
                                  const forwardNums<T> &rhs) {
    return forwardNums<T>(lhs._value + rhs._value, lhs._dv_dx + rhs._dv_dx);
  }

  friend forwardNums<T> operator-(const forwardNums<T> &lhs,
                                  const forwardNums<T> &rhs) {
    return forwardNums<T>(lhs._value - rhs._value, lhs._dv_dx - rhs._dv_dx);
  }

  friend forwardNums<T> operator*(const forwardNums<T> &lhs,
                                  const forwardNums<T> &rhs) {
    return forwardNums<T>(lhs._value * rhs._value,
                          lhs._dv_dx * rhs._value + lhs._value * rhs._dv_dx);
  }

  friend forwardNums<T> operator/(const forwardNums<T> &lhs,
                                  const forwardNums<T> &rhs) {
    return forwardNums<T>(lhs._value / rhs._value,
                          (lhs._dv_dx * rhs._value - lhs._value * rhs._dv_dx) /
                              (rhs._value * rhs._value));
  }
};

} // namespace autodiff
