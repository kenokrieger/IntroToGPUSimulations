/*
 * Simulate the Bornholdt Model.
 */
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include <cmath>
#include <random>

#define timer std::chrono::high_resolution_clock

using std::string; using std::map;


/**
 * @brief Read the entries from a configuration file.
 *
 * Given the path to a configuration file and a delimiter separating keys and values, read in the
 * entries in the file as key value pairs.
 *
 * @param config_filename The path of the configuration file.
 * @param delimiter The delimiter separating keys and values in the configuration file. Default is '='
 *
 * @returns A mapping of key and value pairs as found in the configuration file.
 */
map<string, string> read_config_file(const string& config_filename, const string& delimiter = "=") {
    std::ifstream config_file;
    config_file.open(config_filename);
    map<string, string> config;

    if (!config_file.is_open()) {
        std::cout << "Could not open file '" << config_filename << "'" << std::endl;
        return config;
    }
    int row = 0;
    std::string line;
    std::string key;

    while (getline(config_file, line)) {
        if (line[0] == '#' || line.empty()) continue;
        unsigned long delimiter_position = line.find(delimiter);

        for (int idx = 0; idx < delimiter_position; idx++) {
            if (line[idx] != ' ') key += line[idx];
        }

        std::string value = line.substr(delimiter_position + 1, line.length() - 1);
        config[key] = value;
        row++;
        key = "";
    }
    config_file.close();
    return config;
}


/**
 * @brief Fill an array with entries of -1 and 1.
 *
 * Initialise the values in an array to be either -1 or 1 where the relative amount of
 * -1s is controlled by the parameter 'weight'.
 *
 * @param spins The array to fill.
 * @param grid_height The number of horizontal entries in the array.
 * @param grid_width The number of vertical entries in the array.
 * @param dist Distribution to draw random number from.
 * @param rng The random number generator to use distribution.
 * @param weight The relative amount of -1s in the array. Default is 0.5.
 */
void fill_array(signed char* spins,
                const long long grid_height,
                const long long grid_width,
                std::uniform_real_distribution<float> dist,
                std::mt19937& rng,
                float weight = 0.5f) {
  for (int row = 0; row < grid_height; row++) {
    for (int col = 0; col < grid_width; col++) {
        long long index = row * grid_width + col;
        spins[index] = (dist(rng) < weight) ? -1 : 1;
    }
  }
}


/**
 * @brief Compute the sum of an array.
 *
 * @param arr The array to compute the sum of.
 * @param size The number of elements in the array.
 *
 * @return The sum over the array.
 */
int sum_array(const signed char* arr, unsigned int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
      sum += arr[i];
    }
    return sum;
}


/**
 * @brief Initialise two arrays with values from the set {-1, 1}.
 *
 * Given to arrays assign each entry in the arrays randomly to either -1 or 1.
 *
 * @param black_tiles One of the arrays to be filled.
 * @param white_tiles One of the arrays to be filled.
 * @param grid_width The number of vertical entries in the simulated lattice.
 * @param grid_height The number of horizontal entries in the simulated lattice.
 * @param dist Distribution to draw random number from.
 * @param rng The random number generator to use distribution.
 */
void init_spins(signed char* black_tiles, signed char* white_tiles,
                long long grid_width, long long grid_height,
                std::uniform_real_distribution<float> dist,
                std::mt19937& rng) {
    fill_array(black_tiles, grid_height, grid_width / 2, dist, rng);
    fill_array(white_tiles, grid_height, grid_width / 2, dist, rng);
}


/**
 * @brief Precompute possible spin orientation values.
 *
 * Fill an array with computed values for the spin orientation depending on the
 * spin orientation, its neighbour sum and the simulation parameters.
 *
 * @param probabilities The array to fill with the values.
 * @param market_coupling The relative absolute magnetisation of the system times -2 beta.
 * @param reduced_j The neighbour coupling times -2 beta.
 */
void compute_probabilities(float* probabilities, const double market_coupling, const float reduced_j) {
    for (int idx = 0; idx < 10; idx++) {
      int row = idx / 5;
      int spin = 2 * row - 1;
      double neighbor_sum = -4.0 + 2.0 * (idx - 5 * row);
      double field = reduced_j * neighbor_sum - market_coupling * spin;
      probabilities[idx] = 1 / (1 + exp(field));
    }
}


/**
 * @brief Update the spins in an array according to the dynamics of the Bornholdt Model.
 *
 * Given an array of spins and an array containing the neighbouring spins, for each spin:
 *  1. Compute the neighbours sum
 *  2. Read the spin orientation probability from one of the precomputed values.
 *  3. Change the spin orientation according to its probability.
 *
 * @tparam is_black States whether the spin array contains the white or black tiles according to the
 *                  Checkerboard algorithm.
 * @param spins The array of spins to update.
 * @param neighbour_spins The array containing the neighbour spins.
 * @param probabilities The precomputed probabilities.
 * @param grid_height The number of vertical entries in the array.
 * @param grid_width The number of horizontal entries in the array.
 * @param dist Distribution to draw random number from.
 * @param rng The random number generator to use distribution.
 */
template <bool is_black>
void update_strategies(signed char* spins,
                       const signed char* __restrict__ neighbour_spins,
                       const float* probabilities,
                       const long long grid_height,
                       const long long grid_width,
                       std::uniform_real_distribution<float> dist,
                       std::mt19937& rng) {
  for (int row = 0; row < grid_height; row++) {
    for (int col = 0; col < grid_width; col++) {
      long long index = row * grid_width + col;

      unsigned int lower_neighbor_row = (row + 1) % grid_height;
      unsigned int upper_neighbor_row = (row != 0) ? row - 1: grid_height - 1;
      unsigned int right_neighbor_col = (col + 1) % grid_width;
      unsigned int left_neighbor_col = (col != 0) ? col - 1: grid_width - 1;

      unsigned int horizontal_neighbor_col = right_neighbor_col;
      if (is_black == row % 2)
        horizontal_neighbor_col = left_neighbor_col;

      // Compute sum of nearest neighbour spins
      int neighbour_sum =
          neighbour_spins[upper_neighbor_row * grid_width + col]
        + neighbour_spins[lower_neighbor_row * grid_width + col]
        + neighbour_spins[index]
        + neighbour_spins[row * grid_width + horizontal_neighbor_col];
      float probability = probabilities[5 * (spins[index] + 1) / 2 + (neighbour_sum + 4) / 2];
      spins[index] = (dist(rng) < probability) ? 1 : -1;
    }
  }
}


/**
 * @brief Perform one full lattice update.
 *
 * @param black_tiles One half of the spins on the lattice.
 * @param white_tiles Every other spin on the lattice.
 * @param probabilities Array to be filled with precomputed spin orientation probabilities.
 * @param reduced_alpha The market coupling times -2 beta.
 * @param reduced_j The neighbour coupling times -2 beta.
 * @param grid_height The number of vertical entries in the array.
 * @param grid_width The number of horizontal entries in the array.
 * @param dist Distribution to draw random number from.
 * @param rng The random number generator to use distribution.
 *
 * @return The relative magnetisation of the system before the update.
 */
double update(signed char *black_tiles,
              signed char *white_tiles,
              float* probabilities,
              const float reduced_alpha,
              const float reduced_j,
              const long long grid_height, const long long grid_width,
              std::uniform_real_distribution<float> dist,
              std::mt19937& rng) {
    double magnetisation = sum_array(black_tiles, grid_height * grid_width / 2)
                         + sum_array(white_tiles, grid_height * grid_width / 2);

    double relative_magnetisation = magnetisation / static_cast<double>(grid_width * grid_height);
    double market_coupling = reduced_alpha * fabs(relative_magnetisation);

    compute_probabilities(probabilities, market_coupling, reduced_j);
    update_strategies<true>(
            black_tiles, white_tiles,
            probabilities, grid_height, grid_width / 2,
            dist, rng);
    update_strategies<false>(
            white_tiles, black_tiles,
            probabilities, grid_height, grid_width / 2,
            dist, rng);

    return relative_magnetisation;
}


/**
 * @brief Simulate the Bornholdt Model.
 *
 * @param argc The number of command line arguments.
 * @param argv Array containing the command line arguments.
 * @return The status code of the program execution.
 */
int main(int argc, char** argv) {
  std::ofstream file;
  signed char *black_tiles, *white_tiles;
  float *probabilities;
  string config_filename = (argc == 1) ? "multising.conf" : argv[1];
  map<string, string> config = read_config_file(config_filename);

  const unsigned int grid_height = std::stoll(config["grid_height"]);
  const unsigned int grid_width = std::stoll(config["grid_width"]);
  unsigned int total_updates = std::stoul(config["total_updates"]);
  unsigned int seed = std::stoul(config["seed"]);
  float alpha = std::stof(config["alpha"]);
  float j = std::stof(config["j"]);
  float beta = std::stof(config["beta"]);
  float reduced_alpha = -2.0f * beta * alpha;
  float reduced_j = -2.0f * beta * j;

  // allocate memory for the arrays
  white_tiles = (signed char*) malloc(sizeof(signed char)* grid_width * grid_height / 2);
  black_tiles = (signed char*) malloc(sizeof(signed char) * grid_width * grid_height / 2);
  probabilities = (float *) malloc(10 * sizeof(float));
  // initialise the random number generator
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  init_spins(black_tiles, white_tiles, grid_width, grid_height, dist, rng);

  file.open("magnetisation.dat");
  double magnetisation;
  timer::time_point start = timer::now();
  for (int iteration = 0; iteration < total_updates; iteration++) {
      magnetisation = update(black_tiles, white_tiles, probabilities,
                             reduced_alpha, reduced_j, grid_height, grid_width,
                             dist, rng);
      file << magnetisation << std::endl;
  }
  timer::time_point stop = timer::now();
  file.close();
  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  double spin_updates_per_nanosecond = grid_width * grid_height / duration * 1e-3 * total_updates;
  printf("Total computing time: %f\n", duration * 1e-6);
  printf("Updates per nanosecond: %f\n", spin_updates_per_nanosecond);
  return 0;
}
