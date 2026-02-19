#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mathutils {

enum class NumpyView : uint8_t { Ndarray2D, Ndarray1D };

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
  std::vector<DataType> elements_;
  NumpyView numpy_view_{NumpyView::Ndarray2D};

public:
  using value_type = DataType;

  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), elements_(checked_size(rows, cols)) {}
  Matrix(std::size_t rows, std::size_t cols, NumpyView view)
      : rows_(rows), cols_(cols), elements_(checked_size(rows, cols)),
        numpy_view_(view) {}

  // Matrix(std::size_t rows, std::size_t cols, const DataType &fill)
  //     : rows_(rows), cols_(cols), elements_(checked_size(rows, cols), fill)
  //     {}

  // Construct with explicit storage (must match rows*cols).
  // Matrix(std::size_t rows, std::size_t cols, std::vector<DataType> data)
  //     : rows_(rows), cols_(cols), elements_(std::move(data)) {
  //   if (elements_.size() != checked_size(rows_, cols_)) {
  //     throw std::invalid_argument("Matrix: data size != rows*cols");
  //   }
  // }

  // 2D initializer: Matrix<double> A{{1,2,3},{4,5,6}};
  // Matrix(std::initializer_list<std::initializer_list<DataType>> init) {
  //   rows_ = init.size();
  //   cols_ = rows_ ? init.begin()->size() : 0;

  //   for (const auto &row : init) {
  //     if (row.size() != cols_) {
  //       throw std::invalid_argument("Matrix: ragged initializer_list");
  //     }
  //   }

  //   elements_.reserve(checked_size(rows_, cols_));
  //   for (const auto &row : init) {
  //     elements_.insert(elements_.end(), row.begin(), row.end());
  //   }
  // }

  void set_numpy_view(NumpyView v) { numpy_view_ = v; }
  NumpyView numpy_view() const { return numpy_view_; }
  bool want_numpy_vector() const {
    return (rows_ == 1 || cols_ == 1) && (numpy_view_ == NumpyView::Ndarray1D);
  }

  // Dimensions
  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t size() const noexcept { return elements_.size(); }
  bool empty() const noexcept { return elements_.empty(); }

  // Storage access
  DataType *data() noexcept { return elements_.data(); }
  const DataType *data() const noexcept { return elements_.data(); }

  std::vector<DataType> &vec() noexcept { return elements_; }
  const std::vector<DataType> &vec() const noexcept { return elements_; }

  // Views
  std::span<DataType> span() noexcept {
    return {elements_.data(), elements_.size()};
  }
  std::span<const DataType> span() const noexcept {
    return {elements_.data(), elements_.size()};
  }

  // 2D indexing (unchecked)
  DataType &operator()(std::size_t r, std::size_t c) noexcept {
    return elements_[r * cols_ + c];
  }
  const DataType &operator()(std::size_t r, std::size_t c) const noexcept {
    return elements_[r * cols_ + c];
  }
  // 1D indexing (unchecked)
  DataType &operator[](std::size_t i) noexcept { return elements_[i]; }
  const DataType &operator[](std::size_t i) const noexcept {
    return elements_[i];
  }

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
    return {elements_.data() + r * cols_, cols_};
  }
  std::span<const DataType> row(std::size_t r) const {
    row_bounds_check(r);
    return {elements_.data() + r * cols_, cols_};
  }

  void resize(std::size_t rows, std::size_t cols) {
    rows_ = rows;
    cols_ = cols;
    elements_.assign(checked_size(rows, cols), DataType{});
  }

  void resize(std::size_t rows, std::size_t cols, const DataType &fill) {
    rows_ = rows;
    cols_ = cols;
    elements_.assign(checked_size(rows, cols), fill);
  }

  void fill(const DataType &value) {
    std::fill(elements_.begin(), elements_.end(), value);
  }

  void clear() noexcept {
    rows_ = cols_ = 0;
    elements_.clear();
  }

  DataType maxCoeff() const {
    if (elements_.empty()) {
      throw std::runtime_error("Matrix: maxCoeff() called on empty matrix");
    }
    return *std::max_element(elements_.begin(), elements_.end());
  }

  DataType minCoeff() const {
    if (elements_.empty()) {
      throw std::runtime_error("Matrix: minCoeff() called on empty matrix");
    }
    return *std::min_element(elements_.begin(), elements_.end());
  }

  std::pair<DataType, DataType> minmaxCoeff() const {
    if (elements_.empty()) {
      throw std::runtime_error("Matrix: minmaxCoeff() called on empty matrix");
    }
    auto [min_it, max_it] =
        std::minmax_element(elements_.begin(), elements_.end());
    return {*min_it, *max_it};
  }

  DataType squaredNorm() const {
    DataType norm2{0};
    for (const DataType &x : elements_) {
      norm2 += x * x;
    }
    return norm2;
  }

  template <typename NewDataType>
  Matrix<NewDataType> to_dtype() const
    requires std::is_constructible_v<NewDataType, DataType>
  {
    if constexpr (std::is_same_v<NewDataType, DataType>) {
      return *this;
    } else { // check for overflow/underflow
      const auto [min_val, max_val] = minmaxCoeff();
      const long double new_low =
          static_cast<long double>(std::numeric_limits<NewDataType>::lowest());
      const long double new_hi =
          static_cast<long double>(std::numeric_limits<NewDataType>::max());
      const long double old_min = static_cast<long double>(min_val);
      const long double old_max = static_cast<long double>(max_val);
      if (old_min < new_low || old_max > new_hi) {
        throw std::runtime_error("Matrix: to_dtype would overflow");
      }
      Matrix<NewDataType> out(rows_, cols_, numpy_view_);
      auto *dst = out.data();
      for (std::size_t i = 0; i < elements_.size(); ++i) {
        dst[i] = static_cast<NewDataType>(elements_[i]);
      }
      return out;
    }
  }
};
} // namespace mathutils
