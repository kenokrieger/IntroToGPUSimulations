import numpy as np
from numpy.random import random, randint

# simulation parameters
N = 32
BETA = 1.0
ALPHA = 8.0
J = 1.0
NITERATIONS = 10_000


def main():
    """
    Simulate Bornholdt's spins model of markets.
    """
    mag_record = np.empty((NITERATIONS, ))
    # initialize spins of -1 and 1
    spins = 1 - 2 * randint(0, 2, size=(N, N))
    magnetisation = np.sum(spins) / N ** 2
    for iteration in range(N ** 2 * NITERATIONS):
        # pick a random spin and update it
        row, col = randint(0, N), randint(0, N)
        new_spin = update_spin(spins, row, col, ALPHA, BETA, J, magnetisation)
        magnetisation += (new_spin - spins[row, col]) / N ** 2
        spins[row, col] = new_spin

        # after each "full" lattice update record the magnetisation
        if not iteration % (N ** 2):
            mag_record[iteration // N ** 2] = magnetisation

    np.savetxt("magnetisation.dat", mag_record)
    return 0


def update_spin(spins, row, col, alpha, beta, j, magnetisation):
    """
    Given coordinates of a spin compute the new value and return it.

    Args:
        spins (np.ndarray): An array containing the spins of the simulation.
        row (int): Vertical coordinate of the spin.
        col (int): Horizontal coordinate of the spin.
        alpha (float): Parameter for the magnetisation coupling.
        beta (float): The inverse temperature.
        j (float): Parameter for the neighbour coupling.
        magnetisation (float): The relative magnetisation.

    Returns:
        int: The new spin value according to the Heatbath update.

    """
    neighbour_sum = (
        spins[row - 1, col]
        + spins[(row + 1) % N, col]  # this neat modulo trick ensures boundary conditions
        + spins[row, col - 1]
        + spins[row, (col + 1) % N]
    )
    h = j * neighbour_sum - alpha * spins[row, col] * np.abs(magnetisation)
    if random() < 1 / (1 + np.exp(-2 * beta * h)):
        return 1
    return -1


if __name__ == "__main__":
    main()
