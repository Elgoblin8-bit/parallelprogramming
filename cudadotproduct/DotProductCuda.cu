/*
  Filename   : DotProductCuda.cu
  Author     : Kenner Jimenez
  Assignment : CUDA 
  Description: Using a GPU to attempt a dot product. Currently only accurate until 3000000
*/
/******************************************************************************/
// System includes
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <fmt/format.h>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/universal_vector.h>
#include <vector>

/******************************************************************************/
// Local includes
#include "Timer.hpp"
/******************************************************************************/
// global type aliases
namespace cg = cooperative_groups;
/******************************************************************************/
// Function prototypes
__global__ //
  void
  dotProductCuda (const float* a, const float* b, float* result, int n);

void
fillRandom (thrust::universal_vector<float>& vec);

unsigned
getInput ();

float
dotProductLibrary (const thrust::universal_vector<float>& v1,
                   const thrust::universal_vector<float>& v2);

template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError = 1e-4);


/******************************************************************************/
int
main ()
{

  unsigned N = getInput ();

  thrust::universal_vector<float> vec1 (N);
  thrust::universal_vector<float> vec2 (N);
  thrust::universal_vector<float> result (1, 0.0f);

  fillRandom (vec1);
  fillRandom (vec2);

  Timer t;

  // Config
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  size_t sharedMemSize = (threadsPerBlock / 32) * sizeof (float);

  t.start ();
  dotProductCuda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>> (
    vec1.data ().get (), vec2.data ().get (), result.data ().get (), N);
  cudaDeviceSynchronize ();
  t.stop ();
  double cudaTime = t.getElapsedMs ();

  t.start ();
  float gold = dotProductLibrary (vec1, vec2);
  t.stop ();
  double goldTime = t.getElapsedMs ();
  

  fmt::print ("\nLibrary: {:.4f}\n", result[0]);
  fmt::print ("Time:    {:.4f}\n\n", goldTime);

  fmt::print ("CUDA:    {:.4f}\n", gold);
  fmt::print ("Time:    {:.4f}\n", cudaTime);

  fmt::print ("Speedup: {:.2f}s\n\n", goldTime / cudaTime);
  bool correct = equal (result[0], gold, N);
  fmt::print ("Correct: {}", correct ? "yes" : "NO!");
}
/******************************************************************************/
unsigned
getInput ()
{
  fmt::print ("N ==> ");
  unsigned N;
  std::cin >> N;
  return N;
}
/******************************************************************************/
template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError)
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
/******************************************************************************/
__global__ //
  void
  dotProductCuda (const float* a, const float* b, float* result, int n)
{
  extern __shared__ float sdata[];
  auto block = cg::this_thread_block ();
  auto tile = cg::tiled_partition<32> (block);

  float thread_sum = 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-stride loop
  for (int i = tid; i < n; i += blockDim.x * gridDim.x)
  {
    thread_sum += a[i] * b[i];
  }

  // reduction
  for (int offset = tile.num_threads () / 2; offset > 0; offset /= 2)
  {
    thread_sum += tile.shfl_down (thread_sum, offset);
  }

  if (tile.thread_rank () == 0)
  {
    sdata[threadIdx.x / 32] = thread_sum;
  }
  block.sync ();

  // Reducing
  if (threadIdx.x < blockDim.x / 32)
  {
    float block_sum = sdata[threadIdx.x];
    auto warp0 = cg::tiled_partition<32> (block);
    for (int offset = 16; offset > 0; offset /= 2)
    {
      block_sum += warp0.shfl_down (block_sum, offset);
    }
    // ADDING to a global result
    if (threadIdx.x == 0)
    {
      atomicAdd (result, block_sum);
    }
  }
}
/******************************************************************************/
void
fillRandom (thrust::universal_vector<float>& vec)
{
  std::minstd_rand gen (1);
  std::uniform_real_distribution<float> dis (-4.0f, 4.0f);
  for (auto& val : vec)
  {
    val = dis (gen);
  }
}
/******************************************************************************/
float
dotProductLibrary (const thrust::universal_vector<float>& v1,
                   const thrust::universal_vector<float>& v2)
{
  return std::inner_product (v1.begin (), v1.end (), v2.begin (), 0.0f);
}