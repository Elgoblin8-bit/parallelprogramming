/*
  Author     : Gary M. Zoppetti
  Description: A BLAS multiply for testing correctness.
               Also includes a routine for testing equality between matrices.
*/

#include <algorithm>
#include <limits>
#include <openblas64/cblas.h>
#include <type_traits>

#include "Matrix.hpp"

template<typename T>
void
multiplyBlas (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B)
{
  static_assert (std::is_floating_point_v<T>,
                 "This logic only works for FP types");
  // Use CBLAS general matrix-matrix multiplication
  // C = alpha (A x B) + beta (C)
  if constexpr (std::is_same_v<T, float>)
  {
    cblas_sgemm (CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 A.order (),
                 A.order (),
                 A.order (),
                 1.0f,
                 A.begin (),
                 A.order (),
                 B.begin (),
                 B.order (),
                 0.0f,
                 C.begin (),
                 C.order ());
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    cblas_dgemm (CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 A.order (),
                 A.order (),
                 A.order (),
                 1.0,
                 A.begin (),
                 A.order (),
                 B.begin (),
                 B.order (),
                 0.0,
                 C.begin (),
                 C.order ());
  }
}

template<typename T>
bool
equalMatrices (Matrix<T> const& C1, Matrix<T> const& C2)
{
  auto equalityChecker { [N = C1.order ()] (auto a, auto b)
                         {
                           if constexpr (std::is_floating_point_v<T>)
                           {
                             T absA = std::abs (a);
                             T absB = std::abs (b);
                             T diff = std::abs (a - b);
                             T eps = std::numeric_limits<T>::epsilon ();
                             T tolerance =
                               100 * eps * N *
                               std::max ({ static_cast<T> (1), absA, absB });

                             return diff < tolerance;
                           }
                           else
                             return a == b;
                         } };

  return std::ranges::equal (C1, C2, equalityChecker);
}