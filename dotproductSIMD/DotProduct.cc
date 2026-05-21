/*
  Filename   : FastLane.cc
  Author     : Kenner Jimenez
  Course     : CSMC 476-01
  Date       : 03/23/2021
  Assignment : SIMD Problem
  Description: A dot product program computes the dot product of two vectors
  using intrinsic functions.
*/
/******************************************************************************/
// System includes
#include <algorithm>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <tuple>
#include <vector>

/******************************************************************************/
// Local includes
#include "Timer.hpp"
/******************************************************************************/
// global type aliases
using ParamType = float;
/******************************************************************************/
// Function prototypes
unsigned
getInput ();

std::tuple<std::vector<ParamType>, std::vector<ParamType>>
initializeVectors (auto amount);

template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max, unsigned seed);

void
printResults (std::span<const ParamType> vec1, std::span<const ParamType> vec2);

float
dotProductLibrary (std::span<float const> a, std::span<float const> b);

// Compute parallel scalar product using SIMD.
float
dotProductSimd (std::span<float const> a, std::span<float const> b);

template<std::size_t N, typename F>
void
unroll (F&& f);

template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError = 1e-4);
/******************************************************************************/
int
main ()
{
  unsigned N = getInput ();
  auto [vec1, vec2] = initializeVectors (N);

  Timer timer {};
  timer.start ();
  float resultLib = dotProductLibrary (vec1, vec2);
  timer.stop ();
  double timeLib = timer.getElapsedMs ();

  // testing each step
  print ("vec1: {}\n", vec1);
  print ("vec2: {}\n", vec2);
}
/******************************************************************************/
// Compute parallel scalar product using SIMD.
float
dotProductSimd (std::span<float const> a, std::span<float const> b)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int UNROLL_FACTOR = 2;
  __m256 sum = _mm256_setzero_ps ();
  size_t i {};
  for (; i + VEC_WIDTH <= a.size (); i += VEC_WIDTH)
  {
    __m256 vecA = _mm256_loadu_ps (a.data () + i);
    __m256 vecB = _mm256_loadu_ps (b.data () + i);
    sum = _mm256_fmadd_ps (vecA, vecB, sum);
  }
  // Horizontal add to get the final dot product
  __m256 temp = _mm256_hadd_ps (sum, sum);
  temp = _mm256_hadd_ps (temp, temp);
  float result;
  _mm256_storeu_ps (&result, temp);

  // Handle remaining elements
  for (; i < a.size (); ++i)
    result += a[i] * b[i];

  return result;
}
/******************************************************************************/
// Computing scalar product serially using std::inner_product.
float
dotProductLibrary (std::span<float const> a, std::span<float const> b)
{
  return std::inner_product (a.begin (), a.end (), b.begin (), 0.0f);
}
/******************************************************************************/
// Initializes two vectors with random values between -4.0 and 4.0.
std::tuple<std::vector<ParamType>, std::vector<ParamType>>
initializeVectors (auto amount)
{
  std::vector<ParamType> vec1 (amount);
  fillRandom<float, float> (vec1, -4.0f, 4.0f, 1);
  std::vector<ParamType> vec2 (amount);
  fillRandom<float, float> (vec2, -4.0f, 4.0f, 1);
  return std::tuple { vec1, vec2 };
}
/******************************************************************************/
unsigned
getInput ()
{
  std::print ("N ==> ");
  unsigned N {};
  std::cin >> N;
  return N;
}
/******************************************************************************/
/*
Populates a vector with 'amount' of random integers between 0 and 4 inclusive.
*/
template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max, unsigned seed)
{
  std::minstd_rand randomGenerator (seed);
  std::uniform_real_distribution<float> distribution (min, max);
  std::ranges::generate (seq,
                         [&] () { return distribution (randomGenerator); });
}
/******************************************************************************/
/*
  Author     : Gary M. Zoppetti
  Course     : CMSC 476
  Description: Utility routines.
               Compare two values for equality, taking into account
                 the problem size.
               A loop unroller.
*/
// Embed the code below in your driver.
template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError = 1e-4)
{
  if constexpr (std::floating_point<T>)
  {
    // Handle inf-s, which are problematic when subtracted
    if (a == b)
      return true;
    // Get distance b/w two points
    T diff { std::abs (a - b) };
    // Find largest in absolute value, using "min" for values close to 0
    T max { std::max (
      { std::abs (a), std::abs (b), std::numeric_limits<T>::min () }) };

    return diff < max * std::log2 (N) * relativeError;
  }
  else
    return a == b;
}

// Use this function template for unrolling your loop.
//   You MUST use this.
template<std::size_t N, typename F>
void
unroll (F&& f)
{
  [f]<std::size_t... Is> (std::index_sequence<Is...>)
  {
    // Unrolls into f (0), f (1), ..., f (N - 1)
    (f (std::integral_constant<std::size_t, Is> {}), ...);
  }(std::make_index_sequence<N> {});
}
/******************************************************************************/