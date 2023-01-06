#include <iostream>
#include <fstream>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>

#include "cudamacro.h"
// #include "device/device_reduce.cuh"

#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

enum {C_BLACK, C_WHITE};


/**
 * @brief Contains the different parameters for the simulation
 *
 * @field seed The seed for the random number generator
 * @field reduced_alpha - The parameter alpha multiplied by -2 times beta
 * @field lattice_height - The desired height of the lattice
 * @field lattice_width - The desired width of the lattice
 * @field words_per_row - The number of computer words per row as a result of the chosen configuration
 * @field total_words - The total number of words
 * @field pitch - The pitch of the precomputed probabilities which is needed in the call to cudaMemcpy2D()
 * @field rng_offset - An offset that can be passed to the random number generator to resume a simulation
 *
 */
typedef struct {
    unsigned long long seed;
    float reduced_alpha;
    float reduced_j;
    long long lattice_height;
    long long lattice_width;
    size_t pitch;
    size_t rng_offset;
    float percentage_up;
    dim3 blocks;
    dim3 threads_per_block;
} Parameters;


/**
 * @brief Assign each element in an array randomly (with a weight) to -1 or 1.
 *
 * @tparam COLOR The color of the array in the Checkerboard algorithm.
 * @param seed The seed for the random number generator.
 * @param spins The array to fill with the random numbers.
 * @param grid_height The number of rows in the array.
 * @param grid_width The number of columns in the array.
 * @param weight The relative amount of 1s in the array.
 */
template<int COLOR>
__global__ void fill_array(unsigned int seed,
                           signed char* spins,
                           const long long lattice_height,
                           const long long lattice_width,
                           float weight = 0.5) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  // check for out of bound access
  if ((row >= lattice_height) || (col >= lattice_width)) return;
  // use random number between 0.0 and 1.0 generated beforehand
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, row * lattice_width + col, static_cast<long long>(2 * COLOR), &rng);
  spins[row * lattice_width + col] = (curand_uniform(&rng) < weight) ? 1 : -1;
}


/**
 * @brief Initialise two arrays with randomly distributed -1s and 1s.
 *
 * @param d_black_tiles One of the arrays to fill.
 * @param d_white_tiles One of the arrays to fill.
 * @param params Parameters for filling the arrays such as the number of blocks and threads for
 *               the kernel launch, the lattice dimensions, the seed for the rng and the
 *               initial number of spins pointing up.
 */
void init_traders(signed char* d_black_tiles, signed char* d_white_tiles,
                  Parameters params) {
  fill_array<C_BLACK><<<params.blocks, params.threads_per_block>>>(
          params.seed, d_black_tiles, params.lattice_height, params.lattice_width / 2, params.percentage_up);
  fill_array<C_WHITE><<<params.blocks, params.threads_per_block>>>(
          params.seed, d_white_tiles, params.lattice_height, params.lattice_width / 2, params.percentage_up);
}


/**
 * @brief Precompute the possible spin orientation probabilities
 *
 * @param probabilities  The pointer to the array to be filled with the precomputed probabilities
 * @param market_coupling  The second term in the local field multiplied by -2 times beta times alpha
 *                         market_coupling = -2 * beta * alpha * relative magnetisation
 * @param reduced_j  The parameter j multiplied by -2 times beta
 * @param pitch  The pitch needed for the call to cudaMemcpy2D
 */
void precompute_probabilities(float* probabilities, const float market_coupling, const float reduced_j, const size_t pitch) {
  float h_probabilities[2][5];

  for (int spin = 0; spin < 2; spin++) {
    for (int idx = 0; idx < 5; idx++) {
      int neighbour_sum = 2 * idx - 4;
      float field = reduced_j * neighbour_sum + market_coupling * (-1 + 2 * spin);
      h_probabilities[spin][idx] = 1.0 / (1.0 + exp(field));
    }
  }
  CHECK_CUDA(cudaMemcpy2D(probabilities, 5 * sizeof(*h_probabilities), &h_probabilities, pitch,
                          5 * sizeof(*h_probabilities), 2, cudaMemcpyHostToDevice))
}


/**
Compute the total sum over a given array.

Args:
    d_array: The pointer to the array on the device to compute the sum of.
    size: The number of items in the array.

Returns:
    The value of all the elements summed.
*/
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
long long sum_array(const signed char* d_arr, long long size)
{
  // Reduce
  long long* d_sum;
  int nchunks = (size + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMalloc(&d_sum, nchunks * sizeof(*d_sum)))
  size_t temp_storage_bytes = 0;
  // When d_temp_storage is NULL, no work is done and the required allocation
  // size is returned in temp_storage_bytes.
  void* d_temp_storage = nullptr;
  // determine temporary device storage requirements
  CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_arr, d_sum, CUB_CHUNK_SIZE))
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes))

  for (int i = 0; i < nchunks; i++) {
    CHECK_CUDA(
            cub::DeviceReduce::Sum(
                    d_temp_storage, temp_storage_bytes, &d_arr[i * CUB_CHUNK_SIZE], d_sum + i,
                    std::min((long long) CUB_CHUNK_SIZE, size - i * CUB_CHUNK_SIZE)
            )
    )
  }

  long long* h_sum;
  h_sum = (long long*)malloc(nchunks * sizeof(*h_sum));
  CHECK_CUDA(cudaMemcpy(h_sum, d_sum, nchunks * sizeof(*d_sum), cudaMemcpyDeviceToHost))
  long long total_sum = 0;

  for (int i = 0; i < nchunks; i++) {
    total_sum += h_sum[i];
  }
  CHECK_CUDA(cudaFree(d_sum))
  CHECK_CUDA(cudaFree(d_temp_storage))
  free(h_sum);
  return total_sum;
}


/**
Update the strategy of each trader. The update utilises the Checkerboard
algorithm where traders and their respective neighbors are updated
separately.

template:
    is_black: Specifies which tile color on the checkerboard gets updated.

Args:
    traders: A pointer to the device array of traders.
    checkerboard_agents: The device array containing the neighbors of the
                         traders.
    random_values: A device array containing random float values between 0
                   and 1. Must have the same dimensions as the traders array.
    d_global_market: A pointer to the device integer containing the value of
                     the sum over all traders.
    alpha: A parameter controlling the strength of the market-coupling.
    beta: A parameter controlling the randomness. The greater beta the
          smaller the randomness.
    j: A parameter controlling the strength of the neighbor-coupling.
    grid_height: The height of the grid.
    grid_width: The width of the grid.
    grid_depth: The depth of the grid.
*/
template <bool is_black, int color>
__global__ void update_strategies(const unsigned long long seed, const int rng_invocations,
                                  const unsigned int lattice_height,
                                  const unsigned int lattice_width,
                                  const float precomputed_probabilities[][5],
                                  const signed char* __restrict__ neighbour_spins,
                                  signed char* __restrict__ spins) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  // check for out of bound access
  if (row >= lattice_height || col >= lattice_width) return;

  long long index = row * lattice_width + col;
  int lower_neighbor_row = (row + 1) % lattice_height;
  int upper_neighbor_row = (row != 0) ? row - 1: lattice_height - 1;
  int right_neighbor_col = (col + 1) % lattice_width;
  int left_neighbor_col = (col != 0) ? col - 1: lattice_width - 1;

  int horizontal_neighbor_col = right_neighbor_col;
  if (is_black == row % 2)
    horizontal_neighbor_col = left_neighbor_col;

  int neighbor_sum =
      neighbour_spins[upper_neighbor_row * lattice_width + col]
    + neighbour_spins[lower_neighbor_row * lattice_width + col]
    + neighbour_spins[index]
    + neighbour_spins[row * lattice_width + horizontal_neighbor_col];

  float probability = precomputed_probabilities[5 * ((spins[index] + 1) / 2)][(neighbor_sum + 4) / 2];
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, index, static_cast<long long>(2 * rng_invocations + color),
              &rng);
  spins[index] = curand_uniform(&rng) < probability ? 1 : -1;
}


/**
   Update all of the traders by updating the white and black tiles in succesion.

   Args:
       d_black_tiles: A pointer to the device array containg the black tiles.
       d_white_tiles: A pointer to the device array containing the white tiles.
       random_values: A device array containing/to be filled with random values.
       rng: The generator for the random numbers.
       global_market: The sum over all traders' strategies.
       alpha: A parameter controlling the strength of the market-coupling.
       beta: A parameter controlling the randomness. The greater beta the
             smaller the randomness.
       j: A parameter controlling the strength of the neighbor-coupling.
       grid_height: The height of the grid.
       grid_width: The width of the grid.
       grid_depth: The depth of the grid.

   Returns:
       The sum over all spins.
*/
float update(int iteration,
             signed char *d_black_tiles,
             signed char *d_white_tiles,
             float* d_probabilities,
             Parameters params) {

  long long magnetisation = sum_array(d_black_tiles, params.lattice_height * params.lattice_width / 2);
  magnetisation += sum_array(d_white_tiles, params.lattice_height * params.lattice_width / 2);
  float relative_magnetisation = magnetisation / static_cast<double>(params.lattice_height * params.lattice_width);
  float market_coupling = -params.reduced_alpha * fabs(relative_magnetisation);

  precompute_probabilities(d_probabilities, market_coupling, params.reduced_j, params.pitch);
  CHECK_CUDA(cudaDeviceSynchronize())
  update_strategies<true, C_BLACK><<<params.blocks, params.threads_per_block>>>(
          params.seed, iteration + 1, params.lattice_height, params.lattice_width / 2,
          reinterpret_cast<float (*)[5]>(d_probabilities),
          d_white_tiles, d_black_tiles);
  update_strategies<false, C_WHITE><<<params.blocks, params.threads_per_block>>>(
          params.seed, iteration + 1, params.lattice_height, params.lattice_width / 2,
          reinterpret_cast<float (*)[5]>(d_probabilities),
          d_black_tiles, d_white_tiles);
  return relative_magnetisation;
}
