/*
  Author     : Kenner Jimenez
  TLDR       : Parallel Reduction by forking processes templated
  Description: A program computes the total sum of a set of values.
               The program forks multiple processes to compute partial sums
               and then combines them to get the final result.
*/

/************************************************************/
// System includes
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

/************************************************************/
// Local includes

#include "Timer.hpp"

/************************************************************/
// Concepts

template<typename F, typename T>
concept BinaryFunction = std::regular_invocable<F, T, T> and
                         std::convertible_to<std::invoke_result_t<F, T, T>, T>;

template<typename F, typename T, typename ReturnType>
concept UnaryFunction =
  std::regular_invocable<F, T> and
  std::convertible_to<std::invoke_result_t<F, T>, ReturnType>;

/************************************************************/

template<typename ParamType>
std::vector<ParamType>
populate (auto amount);

template<typename ReturnType>
void
printResults (ReturnType parSum,
              double parTime,
              ReturnType serialSum,
              double serialTime);

template<typename ParamType, typename ReturnType>
ReturnType
transformReducePar (unsigned numProcesses,
                    std::span<ParamType const> v,
                    ReturnType init,
                    BinaryFunction<ReturnType> auto combiner,
                    UnaryFunction<ParamType, ReturnType> auto transformer);

template<typename ParamType, typename ReturnType>
ReturnType
transformReduceOnProc (unsigned id,
                       unsigned numProcesses,
                       std::span<ParamType const> v,
                       BinaryFunction<ReturnType> auto combiner,
                       UnaryFunction<ParamType, ReturnType> auto transformer);

template<typename ParamType, typename ReturnType>
ReturnType
serialTransformReduce (std::span<ParamType const> arrayOfValues,
                       ReturnType init,
                       BinaryFunction<ReturnType> auto combiner,
                       UnaryFunction<ParamType, ReturnType> auto transformer);

/********************************************/
int
main ()
{
  std::printf ("p ==> ");
  int p;
  std::cin >> p;
  std::printf ("(hard concurrency = %d)\n",
               std::thread::hardware_concurrency ());

  std::printf ("\nN ==> ");
  std::string N;
  std::cin >> N;
  std::erase (N, '\'');
  unsigned long nVal = std::stoul (N);
  std::printf ("\n");

  std::vector<char> values = populate<char> (nVal);

  Timer<> timer;
  timer.start ();
  auto sum =
    transformReducePar<char, double> (p,
                                      values,
                                      5.0,
                                      std::plus<double> {},
                                      [&] (char x) { return std::sqrt (x); });
  timer.stop ();
  double parTime = timer.getElapsedMs ();

  timer.start ();
  auto serialSum = serialTransformReduce<char, double> (
    values, 5.0, std::plus<double> {}, [&] (char x) { return std::sqrt (x); });
  timer.stop ();
  double serialTime = timer.getElapsedMs ();

  printResults (sum, parTime, serialSum, serialTime);
}
/************************************************************/
/*
Computing a parallel reduction on a contiguous subrange
of elements of 'v' using process-level parallelism.
Uses 'numProcesses' processes.
*/
template<typename ParamType, typename ReturnType>
ReturnType
transformReducePar (unsigned numProcesses,
                    std::span<ParamType const> v,
                    ReturnType init,
                    BinaryFunction<ReturnType> auto combiner,
                    UnaryFunction<ParamType, ReturnType> auto transformer)
{
  // Create 'numProcesses' pipes for communication between parent and child
  // processes
  std::vector<std::array<int, 2>> pipes (numProcesses);
  for (auto i : std::views::iota (0u, numProcesses))
  {
    pipe (pipes[i].data ());
  }

  // A collection of child process IDs to wait for them later
  std::vector<pid_t> childPids (numProcesses);
  for (auto processId : std::views::iota (0u, numProcesses))
  {
    pid_t childPid = fork ();
    if (childPid == 0)
    {
      // Child process
      // childPids[processId] = getpid ();
      close (pipes[processId][0]);

      ReturnType localSum = transformReduceOnProc<ParamType, ReturnType> (
        processId, numProcesses, v, combiner, transformer);

      write (pipes[processId][1], &localSum, sizeof (ReturnType));
      close (pipes[processId][1]);

      std::exit (0);
    }
    else
    {
      childPids[processId] = childPid;
      // Parent process write end closed immediately.
      close (pipes[processId][1]);
    }
  }

  // Parent process reading sums.
  ReturnType total {};
  for (auto i : std::views::iota (0u, numProcesses))
  {
    ReturnType localSum {};
    read (pipes[i][0], &localSum, sizeof (ReturnType));
    total = combiner (total, localSum);
    close (pipes[i][0]);
  }
  total = combiner (total, init);

  // Wait for all child processes to finish.
  for (auto i : std::views::iota (0u, numProcesses))
  {
    int childStatus {};
    waitpid (childPids[i], &childStatus, 0);
  }
  return total;
}
/************************************************************/
/*
  Compute a reduction on a contiguous subrange of elements of 'v'.
 'id': ID of the process, in the range [0, 'numProcesses').
  Use our partitioning logic so multiple cores run this function
  in parallel.
*/
template<typename ParamType, typename ReturnType>
ReturnType
transformReduceOnProc (unsigned id,
                       unsigned numProcesses,
                       std::span<ParamType const> v,
                       BinaryFunction<ReturnType> auto combiner,
                       UnaryFunction<ParamType, ReturnType> auto transformer)
{
  // Creating a subspan of v from index 'begin' to 'end' using the partitioning
  // math
  auto begin = id * (v.size () / numProcesses) +
               std::min (id, static_cast<unsigned> (v.size () % numProcesses));
  auto end =
    (id + 1) * (v.size () / numProcesses) +
    std::min (id + 1, static_cast<unsigned> (v.size () % numProcesses));

  return std::transform_reduce ((v.begin () + begin),
                                (v.begin () + end),
                                ReturnType {},
                                combiner,
                                transformer);
}

/************************************************************/
/*
  Computing a reduction on the entire vector 'v' using a single process.
*/
template<typename ParamType, typename ReturnType>
ReturnType
serialTransformReduce (std::span<ParamType const> arrayOfValues,
                       ReturnType init,
                       BinaryFunction<ReturnType> auto combiner,
                       UnaryFunction<ParamType, ReturnType> auto transformer)
{
  return std::transform_reduce (
    arrayOfValues.begin (), arrayOfValues.end (), init, combiner, transformer);
}
/************************************************************/
/*
Populates a vector with 'amount' of random integers between 0 and 4 inclusive.
*/
template<typename ParamType>
std::vector<ParamType>
populate (auto amount)
{
  std::minstd_rand randomGenerator (0);

  std::uniform_int_distribution<int> distribution (0, 4);

  std::vector<ParamType> values (amount);
  std::ranges::generate (values,
                         [&] () { return distribution (randomGenerator); });
  return values;
}
/************************************************************/
/*
  Displays the results of the parallel and serial reductions.
*/
template<typename ReturnType>
void
printResults (ReturnType parSum,
              double parTime,
              ReturnType serialSum,
              double serialTime)
{
  std::printf ("// sum:        %.10f\n", parSum);
  std::printf ("// time:       %.2f ms\n\n", parTime);
  std::printf ("Serial sum:    %.10f\n", serialSum);
  std::printf ("Serial time:   %.2f ms\n", serialTime);
}