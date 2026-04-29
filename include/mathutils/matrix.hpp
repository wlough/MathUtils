#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace mathutils {

/**
 * \todo put these to a type_traits.hpp header or something
 */
template <typename T> struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

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

public:
  using value_type = DataType;

  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), elements_(checked_size(rows, cols)) {}
  Matrix(std::size_t rows)
      : rows_(rows), cols_(1), elements_(checked_size(rows, 1)) {}

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

  bool is_vector() const { return (rows_ == 1 || cols_ == 1); }

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
  std::span<DataType> row_span(std::size_t r) {
    row_bounds_check(r);
    return {elements_.data() + r * cols_, cols_};
  }
  std::span<const DataType> row_span(std::size_t r) const {
    row_bounds_check(r);
    return {elements_.data() + r * cols_, cols_};
  }
  Matrix row_copy(std::size_t r) const {
    row_bounds_check(r);
    Matrix out(1, cols_);
    std::copy_n(elements_.data() + r * cols_, cols_, out.data());
    return out;
  }

  Matrix col_copy(std::size_t c) {
    if (c >= cols_) {
      throw std::out_of_range("Matrix: col out of range");
    }
    Matrix<DataType> out(rows_);
    for (std::size_t r = 0; r < rows_; ++r) {
      out.elements_[r] = elements_[r * cols_ + c]; // row-major stride
    }
    return out;
  }

  Matrix transpose() {
    Matrix<DataType> out(cols_, rows_);
    for (std::size_t r = 0; r < rows_; ++r) {
      for (std::size_t c = 0; c < cols_; ++c) {
        out.elements_[c * rows_ + r] =
            elements_[r * cols_ + c]; // row-major stride}
      }
    }
    return out;
  }

  void resize(std::size_t rows, std::size_t cols) {
    rows_ = rows;
    cols_ = cols;
    elements_.assign(checked_size(rows, cols), DataType{});
  }

  void resize(std::size_t rows) { resize(rows, 1); }

  void resize(std::size_t rows, std::size_t cols, const DataType &fill) {
    rows_ = rows;
    cols_ = cols;
    elements_.assign(checked_size(rows, cols), fill);
  }

  void conservativeResize(std::size_t new_rows, std::size_t new_cols) {
    // compute new size (also checks overflow)
    const std::size_t new_size = checked_size(new_rows, new_cols);

    // allocate new storage (value-initialized to DataType{})
    std::vector<DataType> new_elems(new_size, DataType{});

    // copy overlap block (top-left)
    const std::size_t rmin = std::min(rows_, new_rows);
    const std::size_t cmin = std::min(cols_, new_cols);

    for (std::size_t r = 0; r < rmin; ++r) {
      const DataType *src = elements_.data() + r * cols_;
      DataType *dst = new_elems.data() + r * new_cols;
      std::copy(src, src + cmin, dst);
    }

    rows_ = new_rows;
    cols_ = new_cols;
    elements_ = std::move(new_elems);
  }

  void conservativeResize(std::size_t new_rows) {
    conservativeResize(new_rows, 1);
  }

  void fill(const DataType &value) {
    std::fill(elements_.begin(), elements_.end(), value);
  }

  void clear() noexcept {
    rows_ = cols_ = 0;
    elements_.clear();
  }

  void set_row(std::size_t r, std::initializer_list<DataType> xs) {
    row_bounds_check(r);
    if (xs.size() != cols_) {
      throw std::invalid_argument("Matrix::set_row: row " + std::to_string(r) +
                                  " expected " + std::to_string(cols_) +
                                  " values, got " + std::to_string(xs.size()));
    }
    std::copy(xs.begin(), xs.end(), elements_.data() + r * cols_);
  }

  void set_col(std::size_t c, std::initializer_list<DataType> xs) {
    if (c >= cols_) {
      throw std::out_of_range("Matrix: col out of range");
    }
    if (xs.size() != rows_) {
      throw std::invalid_argument("Matrix: set_col wrong length");
    }
    auto it = xs.begin();
    for (std::size_t r = 0; r < rows_; ++r, ++it) {
      elements_[r * cols_ + c] = *it; // row-major stride
    }
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
    DataType norm2 = 0.0;
    for (const auto &x : elements_) {
      const double xd = static_cast<double>(x);
      norm2 += xd * xd;
    }
    return norm2;
  }

  double norm() const { return std::sqrt(squaredNorm()); }

  DataType dot(const Matrix &B) const {
    if (rows_ != B.rows_ || cols_ != B.cols_) {
      throw std::invalid_argument("Matrix::dot: shape mismatch");
    }

    DataType out{}; // zero-init
    for (std::size_t i = 0; i < elements_.size(); ++i) {
      if constexpr (is_std_complex<DataType>::value) {
        out += std::conj(elements_[i]) * B.elements_[i];
      } else {
        out += elements_[i] * B.elements_[i];
      }
    }
    return out;
  }

  template <typename NewDataType>
  Matrix<NewDataType> to_dtype() const
    requires std::is_constructible_v<NewDataType, DataType>
  {
    if constexpr (std::is_same_v<NewDataType, DataType>) {
      return *this;
    } else if (elements_.empty()) {
      Matrix<NewDataType> out(rows_, cols_);
      return out;
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
      Matrix<NewDataType> out(rows_, cols_);
      auto *dst = out.data();
      for (std::size_t i = 0; i < elements_.size(); ++i) {
        dst[i] = static_cast<NewDataType>(elements_[i]);
      }
      return out;
    }
  }

  // Matrix -= Matrix in place subtraction
  Matrix &operator-=(const Matrix &B) {
    if (rows_ != B.rows_ || cols_ != B.cols_) {
      throw std::invalid_argument("Matrix::operator-=: shape mismatch");
    }
    for (std::size_t i = 0; i < elements_.size(); ++i) {
      elements_[i] -= B.elements_[i];
    }
    return *this;
  }

  // Matrix - Matrix subtraction
  Matrix operator-(const Matrix &B) const {
    Matrix out = *this; // copy
    out -= B;           // reuse checked in-place op
    return out;
  }

  // Matrix += Matrix in place addition
  Matrix &operator+=(const Matrix &B) {
    if (rows_ != B.rows_ || cols_ != B.cols_) {
      throw std::invalid_argument("Matrix::operator-=: shape mismatch");
    }
    for (std::size_t i = 0; i < elements_.size(); ++i) {
      elements_[i] += B.elements_[i];
    }
    return *this;
  }

  // Matrix + Matrix addition
  Matrix operator+(const Matrix &B) const {
    Matrix out = *this; // copy
    out += B;           // reuse checked in-place op
    return out;
  }

  // Matrix *= scalar in place multiplication
  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar> &&
             std::is_convertible_v<Scalar, DataType>
  Matrix &operator*=(Scalar a) {
    const DataType aa = static_cast<DataType>(a);
    for (auto &x : elements_) {
      x *= aa;
    }
    return *this;
  }
  // Matrix x scalar multiplication
  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar> &&
             std::is_convertible_v<Scalar, DataType>
  Matrix operator*(Scalar a) const {
    Matrix out = *this;
    out *= a;
    return out;
  }

  // Matrix /= scalar in place division
  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar> &&
             std::is_convertible_v<Scalar, DataType>
  Matrix &operator/=(Scalar a) {
    const DataType aa = static_cast<DataType>(a);
    for (auto &x : elements_) {
      x /= aa;
    }
    return *this;
  }
  // Matrix / scalar division
  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar> &&
             std::is_convertible_v<Scalar, DataType>
  Matrix operator/(Scalar a) const {
    Matrix out = *this;
    out /= a;
    return out;
  }

  // Matrix x Matrix multiplication
  Matrix operator*(const Matrix &B) const {
    if (cols_ != B.rows_) {
      throw std::invalid_argument("Matrix::operator*: shape mismatch");
    }

    Matrix C(rows_, B.cols_);
    // naive O(m*n*k) row-major loop
    for (std::size_t i = 0; i < rows_; ++i) {
      for (std::size_t k = 0; k < cols_; ++k) {
        const DataType aik = (*this)(i, k);
        const std::size_t b_row = k * B.cols_;
        const std::size_t c_row = i * C.cols_;
        for (std::size_t j = 0; j < B.cols_; ++j) {
          C.elements_[c_row + j] += aik * B.elements_[b_row + j];
        }
      }
    }
    return C;
  }
};

// Scalar x Matrix multiplication
template <typename DataType, typename Scalar>
  requires std::is_arithmetic_v<Scalar> &&
           std::is_convertible_v<Scalar, DataType>
Matrix<DataType> operator*(Scalar a, const Matrix<DataType> &A) {
  return A * a;
}

/**
 * @brief Assign an output matrix from a variant holding a matrix of
 * (possibly) different scalar type.
 *
 * This helper extracts the matrix stored in @p v (a @c std::variant of matrix
 * types) and writes it into @p out. If the stored matrix type matches @p
 * OutMat exactly, the assignment is performed directly. Otherwise, if the
 * stored scalar type @c InScalar is constructible as @c OutScalar, the stored
 * matrix is converted to @c OutScalar via @c to_dtype<OutScalar>() and
 * assigned to
 * @p out.
 *
 * If provided, @p key is used only to annotate exception messages with the
 * logical field name (e.g., map/dictionary key) corresponding to @p v.
 *
 * @tparam MatrixVariant Variant type storing matrix objects. Must be a
 * @c std::variant whose alternatives define:
 *   - @c value_type (scalar type),
 *   - @c template <class S> auto to_dtype() const (or compatible) for scalar
 *     conversion to @c S.
 *
 * @tparam OutMat Output matrix type. Must define:
 *   - @c value_type (scalar type),
 *   - copy assignment from @c OutMat,
 *   - @c template <class S> OutMat to_dtype() const (or compatible) for
 * scalar conversion.
 *
 * @param[in]  v    Variant containing an input matrix instance.
 * @param[out] out  Output matrix to be overwritten.
 * @param[in]  key  Optional label used to prefix error messages (typically
 * the map key for @p v). If empty, error messages omit the label.
 *
 * @throws std::runtime_error
 *   - if the stored matrix scalar type cannot be converted to @c OutScalar,
 * or
 *   - if @c to_dtype<OutScalar>() throws (e.g., due to overflow/invalid
 *     conversion checks).
 *
 * @note This function performs a full assignment (and may allocate) depending
 * on @p OutMat. It does not attempt to create a view into the input matrix.
 *
 * @par Example
 * @code{.cpp}
 * using MatrixVariant = std::variant<SamplesField, SamplesIndex,
 * SamplesRGBA>; MatrixVariant v = SamplesField{...}; // e.g., Matrix<float>
 *
 * SamplesField out_f;
 * assign_matrix_from_variant(v, out_f, "X_ambient_V"); // exact dtype or
 * conversion
 *
 * SamplesIndex out_i;
 * assign_matrix_from_variant(v, out_i, "X_ambient_V"); // may throw if
 * incompatible
 * @endcode
 */
template <typename MatrixVariant, class OutMat>
static void assign_matrix_from_variant(const MatrixVariant &v, OutMat &out,
                                       std::string_view key = {}) {
  using OutScalar = typename OutMat::value_type;

  std::visit(
      [&](auto const &in_mat) {
        using InMat = std::decay_t<decltype(in_mat)>;
        using InScalar = typename InMat::value_type;

        if constexpr (std::is_same_v<InMat, OutMat>) {
          out = in_mat; // exact type
        } else if constexpr (std::is_constructible_v<OutScalar, InScalar>) {
          // numeric conversion with overflow checks in to_dtype()
          out = in_mat.template to_dtype<OutScalar>();
        } else {
          throw std::runtime_error(std::string(key) +
                                   " incompatible matrix dtype");
        }
      },
      v);
}

} // namespace mathutils
