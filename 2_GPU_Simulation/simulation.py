from traders import init_traders, update
from datetime import datetime
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from sys import argv

MAX_FILE_SIZE = 100_000


def read_config_file(filename):
    """
    Read the parameters from the configuration file.

    Args:
        filename(str): The location of the configuration file.

    Returns:
        dict: The parameters from the configuration file.

    """
    config = dict()
    with open(filename, 'r') as f:
        args = f.readlines()
        for arg in args:
            if arg == '\n' or arg[0] == '#':
                continue
            else:
                key, value = arg.split("=")
                config[key.strip()] = value.strip()
    return config


def main():
    """Simulate the Bornholdt Model"""
    config = read_config_file("multising.conf" if len(argv) == 1 else argv[1])
    grid_height = int(config["grid_height"])
    grid_width = int(config["grid_width"])
    shape = (grid_height, grid_width // 2)

    black = np.ones(shape, dtype=np.int32)
    d_black = cuda.to_device(black)
    white = np.ones(shape, dtype=np.int32)
    d_white = cuda.to_device(white)

    alpha = float(config["alpha"])
    j = float(config["j"])
    beta = float(config["beta"])
    total_updates = int(config["total_updates"])
    seed = int(config["seed"])
    init_up = float(config["init_up"])
    reduced_alpha = -2 * beta * alpha
    reduced_neighbor_coupling = -2 * beta * j

    threads_per_block = (16, 16)
    blocks = (32, 32)
    total_number_of_threads = (16 ** 2) * (32 ** 2)
    rng_states = create_xoroshiro128p_states(total_number_of_threads, seed=seed)
    magnetisation = np.empty((min(total_updates, MAX_FILE_SIZE), ), dtype=float)
    magnetisation[:] = np.nan

    init_traders[blocks, threads_per_block](rng_states, d_black,
                                            shape, init_up)
    init_traders[blocks, threads_per_block](rng_states, d_white,
                                            shape, init_up)

    save_filename = "magnetisation_0.dat"
    start = datetime.now()
    for iteration in range(total_updates):
        global_market = update(
            rng_states, d_black, d_white, reduced_neighbor_coupling,
            reduced_alpha, shape
        )
        magnetisation[iteration % MAX_FILE_SIZE] = global_market
        if iteration and not (iteration + 1) % MAX_FILE_SIZE:
            np.savetxt(save_filename, magnetisation)
            magnetisation[:] = np.nan
            save_filename = f"magnetisation_{iteration + 1}.dat"

    elapsed_time = (datetime.now() - start)
    cuda.close()
    non_empty_values = np.where(~np.isnan(magnetisation))
    np.savetxt(save_filename,
               magnetisation[non_empty_values])
    flips_per_ns = total_updates * (grid_width * grid_height)
    flips_per_ns /= (elapsed_time.seconds * 1e9
                     + elapsed_time.microseconds * 1e3)
    print("time: {}.{}".format(elapsed_time.seconds, elapsed_time.microseconds))
    print("Spin updates per nanosecond: {:.4E}".format(flips_per_ns))


if __name__ == "__main__":
    main()
