#pragma once

/**
 * @file simple_generator.hpp
 * @brief Simple C++20 coroutine-based generator
  See
 [this](https://www.sparkslabs.com/blog/posts/coroutines-0-modern-cpp-part2-coroutines.html)
 and
 [this](https://github.com/sparkslabs/blog-extras/blob/main/by-date/2023/Coroutines/0/cpp20simple_generator.hpp)

 */

#include <coroutine>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace mathutils {
template <typename T> class GeneratorTemplate {
public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

private:
  handle_type mCoro{}; // owning handle (null means "empty/moved-from")

public:
  /// Construct from coroutine handle (called by the promise).
  explicit GeneratorTemplate(handle_type h) : mCoro(h) {}

  /// Non-copyable (coroutine frames are uniquely owned).
  GeneratorTemplate(const GeneratorTemplate &) = delete;
  GeneratorTemplate &operator=(const GeneratorTemplate &) = delete;

  /// Movable: transfer ownership of the coroutine frame.
  GeneratorTemplate(GeneratorTemplate &&other) noexcept
      : mCoro(std::exchange(other.mCoro, {})) {}

  GeneratorTemplate &operator=(GeneratorTemplate &&other) noexcept {
    if (this != &other) {
      if (mCoro)
        mCoro.destroy();
      mCoro = std::exchange(other.mCoro, {});
    }
    return *this;
  }

  /// Destroy coroutine frame if still owned.
  ~GeneratorTemplate() {
    if (mCoro)
      mCoro.destroy();
  }

  /// True iff this generator owns a coroutine handle.
  bool valid() const noexcept { return static_cast<bool>(mCoro); }

  // -------------------------
  // Coroutine promise interface
  // -------------------------
  struct promise_type {
    T m_current_value{}; // storage for the last yielded value
    std::exception_ptr m_latest_exception{}; // captured exception (if any)

    /// Create the generator object from this promise.
    GeneratorTemplate get_return_object() {
      return GeneratorTemplate{handle_type::from_promise(*this)};
    }

    /// Start suspended: caller controls first resume (via begin()).
    std::suspend_always initial_suspend() noexcept { return {}; }

    /// End suspended so the handle remains valid until destroyed.
    std::suspend_always final_suspend() noexcept { return {}; }

    /// Store the yielded value and suspend.
    std::suspend_always
    yield_value(T v) noexcept(std::is_nothrow_move_assignable_v<T>) {
      m_current_value = std::move(v);
      return {};
    }

    /// No value returned (generator uses `co_yield`, not `co_return value`).
    void return_void() noexcept {}

    /// Capture exceptions to rethrow on the consumer side.
    void unhandled_exception() noexcept {
      m_latest_exception = std::current_exception();
    }
  };

private:
  /// Resume coroutine and rethrow any exception captured in the promise.
  static void resume_and_check(handle_type h) {
    h.resume();
    if (h.promise().m_latest_exception) {
      std::rethrow_exception(h.promise().m_latest_exception);
    }
  }

public:
  // -------------------------
  // Input iterator
  // -------------------------
  class iterator {
    handle_type h{}; // non-owning view of coroutine handle
    bool done{true};

    void settle() noexcept { done = (!h) || h.done(); }

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = const T &;
    using pointer = const T *;

    iterator() = default;

    /// Construct begin iterator: resumes once to reach first `co_yield`.
    explicit iterator(handle_type hh) : h(hh), done(false) {
      if (h) {
        resume_and_check(h); // run until first yield (or completion)
      }
      settle();
    }

    /// Dereference returns a reference to the promise-held current value.
    /// Valid until next increment (which resumes coroutine) or generator death.
    reference operator*() const noexcept { return h.promise().m_current_value; }
    pointer operator->() const noexcept { return &h.promise().m_current_value; }

    /// Pre-increment: resume coroutine to produce the next value.
    iterator &operator++() {
      if (h && !h.done()) {
        resume_and_check(h);
      }
      settle();
      return *this;
    }

    /// Post-increment (standard input-iterator signature).
    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    /// Sentinel-style comparison (sufficient for range-for).
    friend bool operator==(const iterator &a, const iterator &b) noexcept {
      return a.done == b.done;
    }
    friend bool operator!=(const iterator &a, const iterator &b) noexcept {
      return !(a == b);
    }
  };

  /// Begin iteration. If the generator is empty/moved-from, begin==end.
  iterator begin() {
    if (!mCoro)
      return end();
    return iterator{mCoro};
  }

  /// End sentinel iterator.
  iterator end() noexcept { return iterator{}; }
};
} // namespace mathutils

namespace mathutils {

/**
 * @brief Template class for a coroutine that yields values of type T.
 * @tparam T The type of values generated.
 */
template <typename T> class SimpleGenerator {
public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

private:
  handle_type mCoro;

public:
  explicit SimpleGenerator(handle_type h) : mCoro(h) {}

  SimpleGenerator(SimpleGenerator &&other_sg) noexcept : mCoro(other_sg.mCoro) {
    other_sg.mCoro = nullptr;
  }
  SimpleGenerator &operator=(SimpleGenerator &&other) noexcept {
    if (this != other) {
      mCoro = other.mCoro;
      other.mCoro = nullptr;
      return *this;
    }
  }
  SimpleGenerator(const SimpleGenerator &) = delete;
  SimpleGenerator &operator=(const SimpleGenerator &) = delete;
  ~SimpleGenerator() {
    if (mCoro) {
      mCoro.destroy();
    }
  }

  // Implementation of the external API called by the user to actually use the
  // generator
  void start() { try_next(); }
  bool running() { return not mCoro.done(); }
  void try_next() {
    mCoro.resume();
    if (mCoro.promise().m_latest_exception) {
      std::rethrow_exception(mCoro.promise().m_latest_exception);
    }
  }
  T take() { return std::move(mCoro.promise().m_current_value); }

  // Implementation of the internal API called when co_yield/etc are triggered
  // inside the coroutine
  class promise_type {
    T m_current_value;
    std::exception_ptr m_latest_exception;
    friend SimpleGenerator;

  public:
    auto get_return_object() {
      return SimpleGenerator{handle_type::from_promise(*this)};
    }
    auto yield_value(T some_value) {
      m_current_value = some_value; // Capture the yielded value
      return std::suspend_always{};
    }
    auto unhandled_exception() {
      m_latest_exception = std::current_exception();
    }
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto return_void() { return std::suspend_never{}; }
  };

private:
  // Implementation of the iterator protocol
  class iterator {
    SimpleGenerator<T> &owner;
    bool done;
    void iter_next() {
      owner.try_next();
      done = not owner.running();
    }

  public:
    bool operator!=(const iterator &r) const { return done != r.done; }
    auto operator*() const { return owner.take(); }
    iterator &operator++() {
      iter_next();
      return *this;
    }
    iterator(SimpleGenerator<T> &o, bool d) : owner(o), done(d) {
      if (not done)
        iter_next();
    }
  };

public:
  // Public access to the internal iterator protocol

  iterator begin() { return iterator{*this, false}; }
  iterator end() { return iterator{*this, true}; }
};

} // namespace mathutils

/**
 * @example simple_generator.hpp
 *
 * Generate the Fibonacci sequence:
 *
 * @code
 * SimpleGenerator<int> fibs(int max) {
 *   int a{1}, b{1}, n{0};
 *   for (int i = 0; i < max; i++) {
 *     co_yield a;
 *     n = a + b;
 *     a = b;
 *     b = n;
 *   }
 * }
 * @endcode
 *
 *
 * @code
 * int main() {
 *     // Create a generator for Fibonacci numbers
 *     auto generator = fibs(10);
 *
 *     // Use the generator with a range-based for loop
 *     for (int value : generator) {
 *         std::cout << value << " ";
 *     }
 *     std::cout << std::endl;
 *
 *     return 0;
 * }
 * @endcode
 *
 * Generate a sequence of vertices in a clockwise order around a vertex:
 * @code
 * #include "meshbrane/simple_generator.hpp"
 * #include <meshbrane/half_edge_mesh.hpp>
 *
 * using Vert = meshbrane::HalfEdgeVertexBase;
 * SimpleGenerator<Vert> *CCW_vertex_cycle(Vert &v) {
 *  auto h = v.h_out_;
 * auto h_start = h;
 * do {...} while (h != h_start);
 * @endcode
 */