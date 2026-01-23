#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mathutils {

/**
 * @brief A simple 2D matrix class with row-major contiguous storage.
 *
 * @tparam DataType The type of elements stored in the matrix.
 */
template <typename DataType> class Matrix {

private:
  static std::size_t checked_size(std::size_t rows, std::size_t cols) {
    if (cols != 0 && rows > (std::numeric_limits<std::size_t>::max() / cols)) {
      throw std::overflow_error("Matrix: rows*cols overflow");
    }
    return rows * cols;
  }

  void bounds_check(std::size_t r, std::size_t c) const {
    if (r >= rows_ || c >= cols_) {
      throw std::out_of_range("Matrix: index out of range");
    }
  }

  void row_bounds_check(std::size_t r) const {
    if (r >= rows_)
      throw std::out_of_range("Matrix: row out of range");
  }

  std::size_t rows_{0};
  std::size_t cols_{0};
  std::vector<DataType> data_;

public:
  using value_type = DataType;

  Matrix() = default;

  Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(checked_size(rows, cols)) {}

  Matrix(std::size_t rows, std::size_t cols, const DataType &fill)
      : rows_(rows), cols_(cols), data_(checked_size(rows, cols), fill) {}

  // Construct with explicit storage (must match rows*cols).
  Matrix(std::size_t rows, std::size_t cols, std::vector<DataType> data)
      : rows_(rows), cols_(cols), data_(std::move(data)) {
    if (data_.size() != checked_size(rows_, cols_)) {
      throw std::invalid_argument("Matrix: data size != rows*cols");
    }
  }

  // 2D initializer: Matrix<double> A{{1,2,3},{4,5,6}};
  Matrix(std::initializer_list<std::initializer_list<DataType>> init) {
    rows_ = init.size();
    cols_ = rows_ ? init.begin()->size() : 0;

    for (const auto &row : init) {
      if (row.size() != cols_) {
        throw std::invalid_argument("Matrix: ragged initializer_list");
      }
    }

    data_.reserve(checked_size(rows_, cols_));
    for (const auto &row : init) {
      data_.insert(data_.end(), row.begin(), row.end());
    }
  }

  // Dimensions
  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t size() const noexcept { return data_.size(); }
  bool empty() const noexcept { return data_.empty(); }

  // Storage access
  DataType *data() noexcept { return data_.data(); }
  const DataType *data() const noexcept { return data_.data(); }

  std::vector<DataType> &vec() noexcept { return data_; }
  const std::vector<DataType> &vec() const noexcept { return data_; }

  // Views
  std::span<DataType> span() noexcept { return {data_.data(), data_.size()}; }
  std::span<const DataType> span() const noexcept {
    return {data_.data(), data_.size()};
  }

  // 2D indexing (unchecked)
  DataType &operator()(std::size_t r, std::size_t c) noexcept {
    return data_[r * cols_ + c];
  }
  const DataType &operator()(std::size_t r, std::size_t c) const noexcept {
    return data_[r * cols_ + c];
  }
  // 1D indexing (unchecked)
  DataType &operator[](std::size_t i) noexcept { return data_[i]; }
  const DataType &operator[](std::size_t i) const noexcept { return data_[i]; }

  // 2D indexing (checked)
  DataType &at(std::size_t r, std::size_t c) {
    bounds_check(r, c);
    return (*this)(r, c);
  }
  const DataType &at(std::size_t r, std::size_t c) const {
    bounds_check(r, c);
    return (*this)(r, c);
  }

  // Row view (contiguous)
  std::span<DataType> row(std::size_t r) {
    row_bounds_check(r);
    return {data_.data() + r * cols_, cols_};
  }
  std::span<const DataType> row(std::size_t r) const {
    row_bounds_check(r);
    return {data_.data() + r * cols_, cols_};
  }

  void resize(std::size_t rows, std::size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.assign(checked_size(rows, cols), DataType{});
  }

  void resize(std::size_t rows, std::size_t cols, const DataType &fill) {
    rows_ = rows;
    cols_ = cols;
    data_.assign(checked_size(rows, cols), fill);
  }

  void fill(const DataType &value) {
    std::fill(data_.begin(), data_.end(), value);
  }

  void clear() noexcept {
    rows_ = cols_ = 0;
    data_.clear();
  }

  // template <typename NewDataType> Matrix<NewDataType> to_dtype() const {
  //   static_assert(std::is_constructible_v<NewDataType, DataType>,
  //                 "NewDataType must be constructible from DataType");

  //   Matrix<NewDataType> out(rows_, cols_);
  //   auto *dst = out.data();
  //   for (std::size_t i = 0; i < data_.size(); ++i) {
  //     dst[i] = NewDataType(data_[i]);
  //   }
  //   return out;
  // }

  DataType max_coeff() const {
    if (data_.empty()) {
      throw std::runtime_error("Matrix: max_coeff() called on empty matrix");
    }
    return *std::max_element(data_.begin(), data_.end());
  }

  DataType min_coeff() const {
    if (data_.empty()) {
      throw std::runtime_error("Matrix: min_coeff() called on empty matrix");
    }
    return *std::min_element(data_.begin(), data_.end());
  }

  std::pair<DataType, DataType> minmax_coeff() const {
    if (data_.empty()) {
      throw std::runtime_error("Matrix: minmax_coeff() called on empty matrix");
    }
    auto [min_it, max_it] = std::minmax_element(data_.begin(), data_.end());
    return {*min_it, *max_it};
  }

  template <typename NewDataType>
  Matrix<NewDataType> to_dtype() const
    requires std::is_constructible_v<NewDataType, DataType>
  {
    if constexpr (std::is_same_v<NewDataType, DataType>) {
      return *this;
    } else { // check for overflow/underflow
      const auto [min_val, max_val] = minmax_coeff();
      const long double new_low =
          static_cast<long double>(std::numeric_limits<NewDataType>::lowest());
      const long double new_hi =
          static_cast<long double>(std::numeric_limits<NewDataType>::max());
      const long double old_min = static_cast<long double>(min_val);
      const long double old_max = static_cast<long double>(max_val);
      if (old_min < new_low || old_max > new_hi) {
        throw std::runtime_error("Matrix: to_dtype would overflow");
      }
      Matrix<NewDataType> out(rows_, cols_);
      auto *dst = out.data();
      for (std::size_t i = 0; i < data_.size(); ++i) {
        dst[i] = static_cast<NewDataType>(data_[i]);
      }
      return out;
    }
  }
};
} // namespace mathutils