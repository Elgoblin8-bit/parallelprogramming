/*
  Filename   : MatrixMult.cc
  Author     : Kenner Jimenez
  Description: Multiplying two matrices in five different versions.
*/

/************************************************************/
// Runtime Table Avg Results
/*
           Ijk       Jki       Kij     Block  Parallel
1024   4218.98   4620.56   4538.99   1977.34    938.97
1408  11180.15  20977.46  18916.69   5245.74   4191.41
1792  28800.37  47688.97  49836.63  10743.39   5300.85
*/

/************************************************************/
// System includes
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <print>
#include <random>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>

/************************************************************/
// Local includes
#include "Blas.hpp"
#include "Matrix.hpp"
#include "Timer.hpp"

/************************************************************/
// Prototypes

template<typename T, typename U>
void
fillRandom (Matrix<T>& matrix, U min, U max);

template<typename T>
void
multiplyIjk (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B);

template<typename T>
void
multiplyJki (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B);

template<typename T>
void
multiplyKij (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B);

template<typename T>
void
multiplyBlock (Matrix<T>& C,
               Matrix<T> const& A,
               Matrix<T> const& B,
               unsigned blockSize);

template<typename T>
void
multiplyParallel (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B);

std::tuple<unsigned, std::string, unsigned>
getInput ();

void
printResults (unsigned N, std::string version, double time, bool correct);

/************************************************************/
int
main ()
{
  auto [N, version, blockSize] = getInput ();

  Matrix<double> A (N);
  fillRandom (A, -4.0, 4.0);
  Matrix<double> B (N);
  fillRandom (B, -4.0, 4.0);

  Matrix<double> C (N);
  Timer timer {};
  if (version == "ijk")
  {
    timer.start ();
    multiplyIjk<double> (C, A, B);
    timer.stop ();
  }
  else if (version == "jki")
  {
    timer.start ();
    multiplyJki<double> (C, A, B);
    timer.stop ();
  }
  else if (version == "kij")
  {
    timer.start ();
    multiplyKij<double> (C, A, B);
    timer.stop ();
  }
  else if (version == "block")
  {
    timer.start ();
    multiplyBlock<double> (C, A, B, blockSize);
    timer.stop ();
  }
  else if (version == "parallel")
  {
    timer.start ();
    multiplyParallel<double> (C, A, B);
    timer.stop ();
  }

  Matrix<double> blasMatrix (N);
  multiplyBlas (blasMatrix, A, B);
  printResults (
    N, version, timer.getElapsedMs (), equalMatrices<double> (C, blasMatrix));
}
/************************************************************/

/************************************************************/
// Multiply C = A * B using the "block" version of the algorithm.
// L1 use 64 byte blocks
template<typename T>
void
multiplyBlock (Matrix<T>& C,
               Matrix<T> const& A,
               Matrix<T> const& B,
               unsigned blockSize)
{
  unsigned N = A.order ();
  for (unsigned k {}; k < N; k += blockSize)
  {
    for (unsigned i {}; i < N; i += blockSize)
    {
      for (unsigned j {}; j < N; j += blockSize)
      {
        // Initializing and computing the bounds of the
        // mini blocks.
        unsigned iMax = std::min (i + blockSize, N);
        unsigned jMax = std::min (j + blockSize, N);
        unsigned kMax = std::min (k + blockSize, N);

        for (unsigned k2 { k }; k2 < kMax; ++k2)
        {
          for (unsigned i2 { i }; i2 < iMax; ++i2)
          {
            for (unsigned j2 { j }; j2 < jMax; ++j2)
            {
              C[i2, j2] += A[i2, k2] * B[k2, j2];
            }
          }
        }
      }
    }
  }
}
/************************************************************/
// Multiply matrices using the "parallel" version of the algorithm.
// Only parallelizing the outer two loops to prevent false sharing.
template<typename T>
void
multiplyParallel (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B)
{
  unsigned N = A.order ();
  [[omp::directive (parallel for)]]
  for (unsigned i = 0; i < N; ++i)
  {
    [[omp::directive (parallel for)]]
    for (unsigned j = 0; j < N; ++j)
    {
      C[i, j] = 0;
      for (unsigned k = 0; k < N; ++k)
      {
        C[i, j] += A[i, k] * B[k, j];
      }
    }
  }
}
/************************************************************/
// Multiply C = A * B using the "kij" version of the algorithm.
template<typename T>
void
multiplyKij (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B)
{
  unsigned N = A.order ();
  for (unsigned k {}; k < N; ++k)
  {
    for (unsigned i {}; i < N; ++i)
    {
      C[i, k] = 0;
      for (unsigned j {}; j < N; ++j)
      {
        C[i, k] += A[i, j] * B[j, k];
      }
    }
  }
}
/************************************************************/
// Multiply C = A * B using the "jki" version of the algorithm.
template<typename T>
void
multiplyJki (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B)
{
  unsigned N = A.order ();
  for (unsigned j {}; j < N; ++j)
  {
    for (unsigned k {}; k < N; ++k)
    {
      C[k, j] = 0;
      for (unsigned i {}; i < N; ++i)
      {
        C[k, j] += A[i, j] * B[k, i];
      }
    }
  }
}
/************************************************************/
// Multiply C = A * B using the "ijk" version of the algorithm.
template<typename T>
void
multiplyIjk (Matrix<T>& C, Matrix<T> const& A, Matrix<T> const& B)
{
  unsigned N = A.order ();
  for (unsigned i {}; i < N; ++i)
  {
    for (unsigned j {}; j < N; ++j)
    {
      C[i, j] = 0;
      for (unsigned k {}; k < N; ++k)
      {
        C[i, j] += A[i, k] * B[k, j];
      }
    }
  }
}
/************************************************************/
// Prompt and get input.
std::tuple<unsigned, std::string, unsigned>
getInput ()
{
  std::print ("N       ==> ");
  unsigned N {};
  std::cin >> N;

  std::print ("Version ==> ");
  std::string version {};
  std::cin >> version;

  unsigned blockSize {};
  if (version == "block")
  {
    std::print ("B       ==> ");
    std::cin >> blockSize;
  }
  return { N, version, blockSize };
}
/************************************************************/
void
printResults (unsigned N, std::string version, double time, bool correct)
{
  std::print ("\nN       ==> {}", N);
  std::print ("\nVersion ==> {}\n", version);
  std::print ("\nCorrect: {}", correct ? "yes" : "NO!");
  std::print ("\nTime:    {:.2f} ms\n", time);
}
/************************************************************/
// Populate a matrix with values in the range [ min, max ].
template<typename T, typename U>
void
fillRandom (Matrix<T>& matrix, U min, U max)
{
  std::minstd_rand randomGenerator (1);
  if constexpr (std::is_integral_v<T>)
  {
    std::uniform_int_distribution<T> distribution (min, max);
    std::ranges::generate (matrix,
                           [&] () { return distribution (randomGenerator); });
  }
  else if constexpr (std::is_floating_point_v<T>)
  {
    std::uniform_real_distribution<T> distribution (min, max);
    std::ranges::generate (matrix,
                           [&] () { return distribution (randomGenerator); });
  }
}