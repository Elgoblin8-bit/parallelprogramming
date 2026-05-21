/*
  Author     : Gary M. Zoppetti
  Description: Routines to enable parallel transform reductions using threads.
               Includes parallel random number generation and result checking
               routines.
*/

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <execution>
#include <random>
#include <ranges>
#include <span>
#include <type_traits>

/******************************************************************************/

template<typename F, typename ParamType>
concept BinaryFunction =
  std::regular_invocable<F, ParamType, ParamType> and
  std::convertible_to<std::invoke_result_t<F, ParamType, ParamType>, ParamType>;

template<typename F, typename ParamType, typename ReturnType>
concept UnaryFunction =
  std::regular_invocable<F, ParamType> and
  std::convertible_to<std::invoke_result_t<F, ParamType>, ReturnType>;

/******************************************************************************/

struct Range
{
  size_t lower;
  size_t upper;
};

// Compute the subrange of [ 0, 'size' ) that thread 'id'
//   of 'numThreads' should work on.
inline Range
computeRange (unsigned id, unsigned numThreads, size_t size)
{
  size_t lower = (id * size) / numThreads;
  size_t upper = ((id + 1) * size) / numThreads;
  return { lower, upper };
}

/******************************************************************************/

// Fill 'seq' with random numbers in the range [ min, max ].
// Do this in parallel using a std parallel algorithm.
// Use "thread_local" as necessary and also "std::conditional_t"
//   instead of "if constexpr".
template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max)
{
  using Dist = std::conditional_t<std::is_integral_v<T>,
                                  std::uniform_int_distribution<T>,
                                  std::uniform_real_distribution<T>>;
  std::for_each (
    std::execution::par,
    seq.begin (),
    seq.end (),
    [&] (auto& element)
    {
      // Giving each thread their own instance.
      thread_local std::mt19937 generator { std::random_device {}() };
      thread_local Dist dist { static_cast<T> (min), static_cast<T> (max) };
      element = dist (generator);
    });
}

/******************************************************************************/

// Compute a parallel reduction on a contiguous subrange
//   of elements of 'v' using thread-level parallelism.
// Uses 'numThreads' threads.
template<typename ParamType, typename ReturnType>
ReturnType
transformReducePar (unsigned numThreads,
                    std::span<ParamType const> v,
                    ReturnType init,
                    BinaryFunction<ReturnType> auto combiner,
                    UnaryFunction<ParamType, ReturnType> auto transformer)
{
  std::thread mt;
  std::vector<std::thread> threads (numThreads);
  std::vector<ReturnType> partialResults (numThreads);

  for (auto i : std::views::iota (0u, numThreads))
  {
    threads[i] = std::thread (
      [=, &partialResults] ()
      {
        transformReduceOnThread<ParamType, ReturnType> (
          i, numThreads, v, combiner, transformer, partialResults[i]);
      });
  }
  for (auto i : std::views::iota (0u, numThreads))
  {
    threads[i].join ();
  }
  // AT THIS POINT, ALL THREADS SHOULD HAVE FINISHED THEIR WORK AND THE RESULTS
  // ARE IN 'partialResults'.
  for (auto i : std::views::iota (0u, numThreads))
  {
    init = combiner (init, partialResults[i]);
  }
  return init;
}

/******************************************************************************/

// Compute a reduction on a contiguous subrange of elements of 'v'.
// 'id': ID of the thread, in the range [0, 'numThreads').
// Use our partitioning logic so multiple cores run this function
//   in parallel.
// You MAY add parameters at the end.
template<typename ParamType, typename ReturnType>
void
transformReduceOnThread (unsigned id,
                         unsigned numThreads,
                         std::span<ParamType const> v,
                         BinaryFunction<ReturnType> auto combiner,
                         UnaryFunction<ParamType, ReturnType> auto transformer,
                         ReturnType& outsideResult)
{
  Range range = computeRange (id, numThreads, v.size ());
  ReturnType localResult { transformer (v[range.lower]) };

  for (auto i : std::views::iota (range.lower + 1, range.upper))
  {
    localResult = combiner (localResult, transformer (v[i]));
  }
  // Store the result in the location provided by the caller of this function.
  outsideResult = localResult;
}

template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, double relativeError = 1e-6)
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

    return diff < max * relativeError;
  }
  else
    return a == b;
}