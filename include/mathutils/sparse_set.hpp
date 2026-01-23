#pragma once

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace mathutils {

template <typename T> class SparseSet {
  static_assert(std::is_unsigned_v<T>,
                "SparseSet value type T must be an unsigned integer");

private:
  // `dense_` stores elements of the set V
  // For x in V, `sparse_` maps value x -> index i=sparse_[x] of x in `dense_`
  // For x not in V, sparse_[x] is unspecified/stale.
  std::vector<T> dense_;
  std::vector<size_t> sparse_;
  size_t max_size_{0}; // == sparse_.size()

  void erase_at_index_(size_t i) {
    const T x = dense_[i];
    const T last = dense_.back();
    dense_[i] = last;
    sparse_[last] = i;
    dense_.pop_back();
    // sparse_[x] left stale but won't be accessed since x is no longer in the
    // set
  }

public:
  SparseSet() = default;
  SparseSet(std::initializer_list<T> init) {
    for (T v : init)
      insert(v);
  }

  template <typename InputIt> SparseSet(InputIt first, InputIt last) {
    for (auto it = first; it != last; ++it) {
      using V = std::iter_value_t<InputIt>;
      if constexpr (std::is_signed_v<V>) {
        if (*it < 0)
          throw std::invalid_argument(
              "SparseSet constructor: negative value in input range");
      }
      insert(static_cast<T>(*it));
    }
  }

  ///////////////
  // Iterators //
  ///////////////
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  iterator begin() { return dense_.begin(); }
  const_iterator begin() const { return dense_.begin(); }
  const_iterator cbegin() const { return dense_.cbegin(); }

  iterator end() { return dense_.end(); }
  const_iterator end() const { return dense_.end(); }
  const_iterator cend() const { return dense_.cend(); }

  //////////////
  // Capacity //
  //////////////
  bool empty() const { return dense_.empty(); }
  size_t size() const { return dense_.size(); }
  size_t domain_size() const { return max_size_; }

  void clear() { dense_.clear(); }
  void reserve_domain(size_t N) {
    if (N > max_size_) {
      sparse_.resize(N, 0);
      max_size_ = N;
    }
  }

  ////////////
  // Lookup //
  ////////////
  bool contains(T x) const {
    if (x >= max_size_)
      return false;
    const size_t i = sparse_[x];
    return i < dense_.size() && dense_[i] == x;
  }

  ///////////////
  // Modifiers //
  ///////////////
  bool insert(T x) {
    if (contains(x))
      return false;
    if (x >= max_size_)
      reserve_domain(static_cast<size_t>(x) + 1);
    sparse_[x] = dense_.size();
    dense_.push_back(x);
    return true;
  }

  bool erase(T x) {
    if (!contains(x))
      return false;

    const size_t i = sparse_[x];
    const T last = dense_.back();

    dense_[i] = last;
    sparse_[last] = i;
    dense_.pop_back();
    return true;
  }
  ///////////////
  // Relations //
  ///////////////
  // subset
  bool operator<=(const SparseSet &other) const {
    for (T x : dense_)
      if (!other.contains(x))
        return false;
    return true;
  }
  // proper subset
  bool operator<(const SparseSet &other) const {
    if (size() >= other.size())
      return false;
    for (T x : dense_)
      if (!other.contains(x))
        return false;
    return true;
  }
  // equality (as sets)
  bool operator==(const SparseSet &other) const {
    if (size() != other.size())
      return false;
    return (*this <= other);
  }
  bool operator!=(const SparseSet &other) const { return !(*this == other); }

  ////////////////////////
  // Set ops (in-place) //
  ////////////////////////
  // intersection
  SparseSet &operator&=(const SparseSet &other) {
    for (size_t i = 0; i < dense_.size();) {
      if (!other.contains(dense_[i]))
        erase_at_index_(i);
      else
        ++i;
    }
    return *this;
  }
  // union
  SparseSet &operator|=(const SparseSet &other) {
    for (T x : other)
      insert(x);
    return *this;
  }
  // difference
  SparseSet &operator-=(const SparseSet &other) {
    for (size_t i = 0; i < dense_.size();) {
      if (other.contains(dense_[i])) {
        erase_at_index_(
            i); // do not increment i: new element swapped into slot i
      } else {
        ++i;
      }
    }
    return *this;
  }

  ////////////////////////////
  // Set ops (non-mutating) //
  ////////////////////////////
  // union
  friend SparseSet operator|(SparseSet a, const SparseSet &b) {
    a |= b;
    return a;
  }
  // intersection
  friend SparseSet operator&(SparseSet a, const SparseSet &b) {
    a &= b;
    return a;
  }
  // difference
  friend SparseSet operator-(SparseSet a, const SparseSet &b) {
    a -= b;
    return a;
  }
};

} // namespace mathutils
