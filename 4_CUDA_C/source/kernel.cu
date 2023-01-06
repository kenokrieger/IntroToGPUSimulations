#include <iostream>
#include <chrono>
#include <map>

#include "traders.cuh"
#include "cudamacro.h"

using std::string; using std::map;

#define timer std::chrono::high_resolution_clock
#define THREADS_X (16)
#define THREADS_Y (16)
#define DIV_UP(a,b)  (((a) + ((b) - 1)) / (b))


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


void validate_grid(const long long lattice_width, const long long lattice_height) {
  if (!lattice_width || (lattice_width % 2) || ((lattice_width / 2) % THREADS_X)) {
    fprintf(stderr, "\nPlease specify an lattice_width multiple of %d\n\n",
            2 * THREADS_X);
    exit(EXIT_FAILURE);
  }
  if (!lattice_height || (lattice_height % THREADS_Y)) {
    fprintf(stderr, "\nPlease specify a lattice_height multiple of %d\n\n", THREADS_Y);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Run a GPU simulation of the Bornholdt Model.
 *
 * @param argc Number of command line arguments.
 * @param argv Array containing the command line arguments.
 *
 * @return The status code of the program.
 */
int main(int argc, char** argv) {
  signed char *d_black_tiles, *d_white_tiles;
  float *d_probabilities;
  Parameters params;

  const char *config_filename = (argc == 1) ? "multising.conf" : argv[1];
  map<string, string> config = read_config_file(config_filename);
  params.lattice_height = std::stoll(config["lattice_height"]);
  params.lattice_width = std::stoll(config["lattice_width"]);
  params.seed = std::stoull(config["seed"]);
  const unsigned int total_updates = std::stoul(config["total_updates"]);
  float alpha = std::stof(config["alpha"]);
  float j = std::stof(config["j"]);
  float beta = std::stof(config["beta"]);
  if (config.count("init_up")) {
    params.percentage_up = std::stoull(config["init_up"]);
  } else {
    params.percentage_up = 0.5;
  }
  if (config.count("rng_offset")) {
    params.rng_offset = std::stoull(config["rng_offset"]);
  } else {
    params.rng_offset = 0;
  }
  params.reduced_alpha = -2.0f * beta * alpha;
  params.reduced_j = -2.0f * beta * j;
  validate_grid(params.lattice_width, params.lattice_height);
  params.blocks = (DIV_UP(params.lattice_width / 2, THREADS_X),
                   DIV_UP(params.lattice_height, THREADS_Y));
  params.threads_per_block = (THREADS_X, THREADS_Y);
  // allocate memory for the arrays
  CHECK_CUDA(cudaMalloc(&d_white_tiles, params.lattice_height * params.lattice_width / 2 * sizeof(*d_white_tiles)))
  CHECK_CUDA(cudaMalloc(&d_black_tiles, params.lattice_height * params.lattice_width / 2 * sizeof(*d_black_tiles)))
  CHECK_CUDA(cudaMallocPitch(&d_probabilities, &params.pitch, 5 * sizeof(*d_probabilities), 2))
  init_traders(d_black_tiles, d_white_tiles, params);
  // Synchronize operations on the GPU with CPU
  CHECK_CUDA(cudaDeviceSynchronize())

  std::ofstream file;
  file.open("magnetisation.dat");
  timer::time_point start = timer::now();
  for (int iteration = 0; iteration < total_updates; iteration++) {
      float relative_magnetisation = update(iteration, d_black_tiles, d_white_tiles, d_probabilities, params);
      file << relative_magnetisation << std::endl;
  }
  timer::time_point stop = timer::now();
  file.close();

  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  double spin_updates_per_nanosecond = total_updates * params.lattice_width * params.lattice_height / duration * 1e-3;
  printf("Total computing time: %f\n", duration * 1e-6);
  printf("Updates per nanosecond: %f\n", spin_updates_per_nanosecond);
  return 0;
}
