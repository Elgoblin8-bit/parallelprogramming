/*
  Filename   : ParallelReduce.cc
  Author     : Kenner Jimenez
  Assignment : Parallel Reduction 
  Description: A sum program computes the total sum of a set of values.
  Each thread calculates a local sum over its own chunk of data,
  and then all thread sums are accumulated in the master thread
*/
/******************************************************************************/
// System includes
#include <cmath>
#include <iostream>
#include <numeric>
#include <print>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

/******************************************************************************/
// Local includes
#include "ParallelReduce.hpp"
#include "Timer.hpp"

/******************************************************************************/
// global type aliases
using ParamType = char;
using ReturnType = double;

/******************************************************************************/
// Function prototypes

std::tuple<unsigned, unsigned long>
getInput ();

void
calculateResults (unsigned numThreads, unsigned long nValue);

std::tuple<ReturnType, ReturnType>
calculateSerialResults (unsigned numThreads,
                        unsigned long nValue,
                        std::vector<ParamType>& sequence);

std::tuple<ReturnType, ReturnType>
calculateStandardParallelResults (unsigned numThreads,
                                  unsigned long nValue,
                                  std::vector<ParamType>& sequence);
void
printResults (ReturnType parallelSum,
              ReturnType standardParallelSum,
              ReturnType serialSum,
              ReturnType parallelTime,
              ReturnType standardParallelTime,
              ReturnType serialTime);

/******************************************************************************/
int
main ()
{
  auto [threads, nValue] = getInput ();

  calculateResults (threads, nValue);
}
/******************************************************************************/
void
calculateResults (unsigned numThreads, unsigned long nValue)
{

  // Starting with my implementation
  // Initializing
  std::vector<ParamType> sequence (nValue);
  fillRandom (std::span { sequence }, 0ul, 4ul);
  Timer timer {};

  timer.start ();
  auto parallelSum = transformReducePar<ParamType, ReturnType> (
    numThreads,
    sequence,
    5.0,
    std::plus<ReturnType> {},
    [] (ParamType x) { return std::sqrt (static_cast<ReturnType> (x)); });
  timer.stop ();
  ReturnType parallelTime = timer.getElapsedMs ();

  // Standard library parallel algorithm
  auto [standardParallelSum, standardParallelTime] =
    calculateStandardParallelResults (numThreads, nValue, sequence);

  // Calling the serial version here instead of in main() to keep it clean
  auto [serialSum, serialTime] =
    calculateSerialResults (numThreads, nValue, sequence);

  printResults (parallelSum,
                standardParallelSum,
                serialSum,
                parallelTime,
                standardParallelTime,
                serialTime);
}
/******************************************************************************/
// Compute a parallel reduction on a contiguous subrange. With the standard
// parallel reduce algorithm,only get the final result, but with
// implementation, can also get the partial results from each thread.
std::tuple<ReturnType, ReturnType>
calculateStandardParallelResults (unsigned numThreads,
                                  unsigned long nValue,
                                  std::vector<ParamType>& sequence)
{
  Timer timer {};
  timer.start ();
  auto standardParallelSum = std::transform_reduce (
    std::execution::par,
    sequence.begin (),
    sequence.end (),
    5.0,
    std::plus<ReturnType> {},
    [] (ParamType x) { return std::sqrt (static_cast<ReturnType> (x)); });

  timer.stop ();
  ReturnType standardParallelTime = timer.getElapsedMs ();

  return { standardParallelSum, standardParallelTime };
}
/******************************************************************************/
// Compute a parallel reduction on a contiguous subrange serially with one
// thread. Helps verify the correctness of the parallel version.
std::tuple<ReturnType, ReturnType>
calculateSerialResults (unsigned numThreads,
                        unsigned long nValue,
                        std::vector<ParamType>& sequence)
{
  Timer timer {};
  timer.start ();

  auto serialSum = std::transform_reduce (
    std::execution::seq,
    sequence.begin (),
    sequence.end (),
    5.0,
    std::plus<ReturnType> {},
    [] (ParamType x) { return std::sqrt (static_cast<ReturnType> (x)); });
  timer.stop ();
  ReturnType serialTime = timer.getElapsedMs ();

  return { serialSum, serialTime };
}
/******************************************************************************/
void
printResults (ReturnType parallelSum,
              ReturnType standardParallelSum,
              ReturnType serialSum,
              ReturnType parallelTime,
              ReturnType standardParallelTime,
              ReturnType serialTime)
{
  std::print ("// sum:       {}\n", parallelSum);
  std::print ("// time:      {:.2f}ms\n", parallelTime);
  std::print ("\n// alg sum:   {}\n", standardParallelSum);
  std::print ("// alg time:  {:.2f}ms\n", standardParallelTime);
  std::print ("\nSerial sum:   {}\n", serialSum);
  std::print ("Serial time:  {:.2f}ms\n", serialTime);

  std::print ("\nCorrect:      {}\n",
              equal (parallelSum, serialSum) ? "yes" : "NO!");
}
/******************************************************************************/
// Get the number of threads and the value of N from the user.
std::tuple<unsigned, unsigned long>
getInput ()
{
  std::print ("(hard concurrency = {})\n",
              std::thread::hardware_concurrency ());
  std::print ("p ==> ");
  unsigned p;
  std::cin >> p;
  std::print ("\nN ==> ");
  std::string N;
  std::cin >> N;
  std::erase (N, '\'');
  unsigned long nValue = std::stoul (N);
  std::print ("\n");

  return { p, nValue };
}