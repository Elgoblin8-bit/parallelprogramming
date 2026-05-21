/*
  Filename   : Matrix.hpp
  Author     : Kenner Jimenez
  Description: Class for representing square matrices of order
               N (i.e., N x N).
*/

/************************************************************/
// Prevent multiple inclusion

#pragma once

/************************************************************/
// System includes

#include <algorithm>
#include <memory>
#include <new>
#include <openblas64/cblas.h>

/************************************************************/

template<typename T>
class Matrix
{
public:
  using iterator = T*;
  using const_iterator = T const*;

  /**********************************************************/

  // Delete default ctor. User should always specify a size.
  Matrix () = delete;

  /**********************************************************/

  // Initialize a square matrix of order 'order'.
  Matrix (unsigned order)
      : m_data (new (CACHE_LINE_BYTES) T[order * order]),
        m_order (order)
  {
  }

  /**********************************************************/

  // Copy ctor.
  // Use a delegating ctor.
  Matrix (Matrix const& m)
      : Matrix (m.order ())
  {
    std::copy (m.begin (), m.end (), begin ());
  }

  /**********************************************************/

  // Move ctor.
  Matrix (Matrix&& m) noexcept
      : Matrix (m.order ())
  {
    std::move (m.begin (), m.end (), begin ());
  }

  /**********************************************************/

  // Dtor. Default is fine thanks to std::unique_ptr!
  ~Matrix () = default;

  /**********************************************************/

  // Swap matrix objects. Use in other methods.
  void
  swap (Matrix& other) noexcept
  {
    using std::swap;
    swap (m_data, other.m_data);
    swap (m_order, other.m_order);
  }

  /**********************************************************/

  // Copy assignment.
  // Use copy swap idiom.
  Matrix&
  operator= (Matrix m)
  {
    swap (m);
    return *this;
  }

  /**********************************************************/

  // Move assignment.
  // Use copy swap idiom.
  Matrix&
  operator= (Matrix&& m) noexcept
  {
    swap (m);
    return *this;
  }

  /**********************************************************/

  // Return the appropriate element
  // Do NOT do bounds checking
  T&
  operator[] (unsigned row, unsigned col)
  {
    return m_data[row * m_order + col];
  }

  /**********************************************************/

  // Return the appropriate element
  // Do NOT do bounds checking
  T const&
  operator[] (unsigned row, unsigned col) const
  {
    T const& element = m_data[row * m_order + col];
    return element;
  }

  /**********************************************************/

  // Return the order
  unsigned
  order () const
  {
    return m_order;
  }

  /**********************************************************/

  // Return the number of elements
  size_t
  size () const
  {
    return m_order * m_order;
  }

  /**********************************************************/

  // Return pointer to first element
  iterator
  begin ()
  {
    return m_data.get ();
  }

  /**********************************************************/

  // Return pointer to first element
  const_iterator
  begin () const
  {
    const_iterator iter { m_data.get () };
    return iter;
  }

  /**********************************************************/

  // Return pointer to one beyond last element
  iterator
  end ()
  {
    return m_data.get () + size ();
  }

  /**********************************************************/

  // Return pointer to one beyond last element
  const_iterator
  end () const
  {
    const_iterator iter { m_data.get () + size () };
    return iter;
  }

  /**********************************************************/

private:
  static constexpr std::align_val_t CACHE_LINE_BYTES { 64 };

private:
  std::unique_ptr<T[]> m_data;
  unsigned m_order;
};

/************************************************************/