#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

/************************************************************/
// Local includes
#include "ParallelReduce.hpp"

/************************************************************/

TEST_CASE ("Random Generation")
{
  SUBCASE ("Integers")
  {
    std::span<int> seq (new int[25], 25);
    fillRandom (seq, 0, 4);
    for (int element : seq)
      CHECK (element >= 0);
    for (int element : seq)
      CHECK (element <= 4);
  }
  SUBCASE ("Doubles")
  {
    std::span<double> seq (new double[25], 25);
    fillRandom (seq, 0.0, 4.0);
    for (double element : seq)
      CHECK (element >= 0.0);
    for (double element : seq)
      CHECK (element <= 4.0);
  }
}
TEST_CASE ("Computing Ranges")
{
  SUBCASE ("Even division")
  {
    // 100 elements across 4 threads should split evenly  25 each
    CHECK (computeRange (0, 4, 100).lower == 0);
    CHECK (computeRange (0, 4, 100).upper == 25);
    CHECK (computeRange (1, 4, 100).lower == 25);
    CHECK (computeRange (1, 4, 100).upper == 50);
    CHECK (computeRange (2, 4, 100).lower == 50);
    CHECK (computeRange (2, 4, 100).upper == 75);
    CHECK (computeRange (3, 4, 100).lower == 75);
    CHECK (computeRange (3, 4, 100).upper == 100);
  }

  SUBCASE ("Uneven division")
  {
    // 10 elements across 3 threads won't split evenly
    CHECK (computeRange (0, 3, 10).lower == 0);
    CHECK (computeRange (0, 3, 10).upper == 3);
    CHECK (computeRange (1, 3, 10).lower == 3);
    CHECK (computeRange (1, 3, 10).upper == 6);
    CHECK (computeRange (2, 3, 10).lower == 6);
    CHECK (computeRange (2, 3, 10).upper == 10);
  }

  SUBCASE ("Single thread gets everything")
  {
    CHECK (computeRange (0, 1, 100).lower == 0);
    CHECK (computeRange (0, 1, 100).upper == 100);
  }

  SUBCASE ("Ranges are contiguous and non-overlapping")
  {
    // each thread's upper should equal the next thread's lower
    unsigned numThreads = 4;
    size_t size = 100;
    for (unsigned i = 0; i < numThreads - 1; ++i)
      CHECK (computeRange (i, numThreads, size).upper ==
             computeRange (i + 1, numThreads, size).lower);
  }

  SUBCASE ("Last thread upper equals size")
  {
    CHECK (computeRange (3, 4, 100).upper == 100);
    CHECK (computeRange (2, 3, 10).upper == 10);
  }
}
TEST_CASE ("transformReducePar")
{
  SUBCASE ("Integer sum")
  {
    std::vector<int> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::span<int const> v (data);
    int result = transformReducePar<int, int> (
      4, v, 0, [] (int a, int b) { return a + b; }, [] (int x) { return x; });
    CHECK (result == 55);
  }

  SUBCASE ("Double product")
  {
    std::vector<double> data = { 1.0, 2.0, 3.0, 4.0 };
    std::span<double const> v (data);
    double result = transformReducePar<double, double> (
      2,
      v,
      1.0,
      [] (double a, double b) { return a * b; },
      [] (double x) { return x; });
    CHECK (result == 24.0);
  }
  SUBCASE ("Integer sum with transform")
  {
    // square each element then sum
    std::vector<int> data = { 1, 2, 3, 4, 5 };
    std::span<int const> v (data);
    int result = transformReducePar<int, int> (
      4,
      v,
      0,
      [] (int a, int b) { return a + b; },
      [] (int x) { return x * x; });
    CHECK (result == 55);
  }
  SUBCASE ("Double sum with transform")
  {
    // sqrt each element then sum
    std::vector<double> data = { 1.0, 4.0, 9.0, 16.0, 25.0 };
    std::span<double const> v (data);
    double result = transformReducePar<double, double> (
      4,
      v,
      0.0,
      [] (double a, double b) { return a + b; },
      [] (double x) { return std::sqrt (x); });
    CHECK (equal (result, 15.0));
  }
  SUBCASE ("Not zero init")
  {
    std::vector<int> data = { 1, 2, 3, 4, 5 };
    std::span<int const> v (data);
    int result = transformReducePar<int, int> (
      4, v, 10, [] (int a, int b) { return a + b; }, [] (int x) { return x; });
    CHECK (result == 25);
  }
}