#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

/*
  Filename   : reduction.cc
  Author     : Kenner Jimenez
  Course     : CSMC 476-01
  Date       : 01/10/2021
  Assignment : Reduction Problem
  Description: A sum program computes the total sum of a set of values.
  Each 'processor' first calculates a local sum over its own chunk of data,
  and then all local sums are combined into a single result.
*/

void
populate (std::vector<int>& array,
          std::vector<int>& accumlator,
          int processors);

int
reduction (std::vector<int>& initial,
           std::vector<int>& accumlator,
           int processors);

void
printTable (std::vector<int>& firstV,
            std::vector<int>& finalV,
            int totalStages,
            int totalAdditions);

int
main ()
{
  /*
    Initializing all variables and vectors for the problem and then getting user
    input for p.
  */
  int p, stages, adds;
  std::vector<int> A {}, S {};
  std::printf ("p ==> ");
  std::cin >> p;
  std::printf ("\n");

  // after a some testing the amount of adds is always p - 1
  adds = p - 1;

  populate (A, S, p);

  stages = reduction (A, S, p);

  printTable (S, A, stages, adds);
}

/************************************************************/

/*
  Populates the vector 'array' with random numbers and the accumlator with
  copies of 'array'
*/
void
populate (std::vector<int>& array, std::vector<int>& accumlator, int processors)
{
  // Creates a random number generator with the seed 0
  std::minstd_rand randomGenerator (0);

  // Defining the range of the random numbers
  std::uniform_int_distribution<int> distribution (0, 99);

  // Filling the vectors
  for (int i {}; i < processors; ++i)
  {
    array.push_back (distribution (randomGenerator));
  }

  /*
    Copying the contents of array into accumlator with back_inserter since
    accumlator is empty at this point and it seg faults otherwise (very
    annoying).
  */
  std::copy (array.begin (), array.end (), std::back_inserter (accumlator));
}

/************************************************************/

/*
  Performing the reduction operation on the accumlator vector
*/
int
reduction (std::vector<int>& initial,
           std::vector<int>& accumlator,
           int processors)
{
  // Initializing variables
  // Will populate a small vector with the stages and adds for printing later
  int stages = 0;

  // The starting distance between receiving 'processors'
  int distance = 2;

  // The starting gap between sending and receiving 'processors'
  int gap = 1;

  for (int s {}; gap < processors; ++s)
  {

    std::printf ("Stage %d \n--------\n", s);

    for (int i {}; i < processors; ++i)
    {
      if (i % distance == 0)
      {
        accumlator[i] += accumlator[i + gap];
        /*
          Printing the necessary information for each addition within each stage
        */
        std::printf ("Recv: %d from %d, v = %d, sum = %d\n\n",
                     i,
                     (i + gap),
                     accumlator[i + gap],
                     accumlator[i]);
      }
    }

    distance *= 2;
    gap *= 2;
    stages++;
  }

  return stages;
}

/************************************************************/

/*
  Printing the table with each stage like shown in document Recv: P1 from P2,
  value = #, sum = #
*/
void
printTable (std::vector<int>& firstV,
            std::vector<int>& finalV,
            int totalStages,
            int totalAdditions)
{

  std::printf ("Summary\n======= \n");

  std::printf ("A[] = [");
  bool first = true;
  for (int v : firstV)
  {
    if (!first)
      std::printf (", ");
    std::printf ("%d", v);
    first = false;
  }
  std::printf ("]\n");

  std::printf ("S[] = [");
  first = true;
  for (int v : finalV)
  {
    if (!first)
      std::printf (", ");
    std::printf ("%d", v);
    first = false;
  }

  std::printf ("]\n");

  std::printf ("# stages = %d\n", totalStages);

  std::printf ("# adds   = %d\n", totalAdditions);
}