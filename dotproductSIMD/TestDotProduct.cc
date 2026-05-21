#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

/************************************************************/
// System includes
#include <span>
#include <vector>
/************************************************************/
// Local includes
#define TESTING
#include "DotProduct.cc"
/************************************************************/

/******************************************************************************/

TEST_CASE ("empty vectors return 0")
{
  std::vector<float> a, b;
  CHECK (dotProductSimd (a, b) == doctest::Approx (0.f));
  CHECK (dotProductLibrary (a, b) == doctest::Approx (0.f));
}

TEST_CASE ("single element")
{
  std::vector<float> a = { 3.f };
  std::vector<float> b = { 4.f };
  CHECK (dotProductSimd (a, b) == doctest::Approx (12.f).epsilon (1e-4));
  CHECK (dotProductLibrary (a, b) == doctest::Approx (12.f).epsilon (1e-4));
}

TEST_CASE ("all ones — result equals N")
{
  for (size_t n : { 1u, 7u, 8u, 9u, 15u, 16u, 17u, 32u, 33u })
  {
    std::vector<float> a (n, 1.f), b (n, 1.f);
    INFO ("n = " << n);
    CHECK (dotProductSimd (a, b) == doctest::Approx ((float) n).epsilon (1e-4));
    CHECK (dotProductLibrary (a, b) ==
           doctest::Approx ((float) n).epsilon (1e-4));
  }
}

TEST_CASE ("all zeros — result is 0")
{
  std::vector<float> a (64, 0.f), b (64, 0.f);
  CHECK (dotProductSimd (a, b) == doctest::Approx (0.f));
  CHECK (dotProductLibrary (a, b) == doctest::Approx (0.f));
}

TEST_CASE ("negative values")
{
  std::vector<float> a = { -1.f, -2.f, -3.f, -4.f };
  std::vector<float> b = { 1.f, 2.f, 3.f, 4.f };
  CHECK (dotProductSimd (a, b) == doctest::Approx (-30.f).epsilon (1e-4));
  CHECK (dotProductLibrary (a, b) == doctest::Approx (-30.f).epsilon (1e-4));
}