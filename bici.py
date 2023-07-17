import itertools as it

import matplotlib.pyplot as plt

import numpy as np


def initializate_lattice(size, ordered=True):
    return (
        np.ones((size, size))
        if ordered
        else np.random.choice([1, -1], size=(size, size))
    )


def statistics(spins):
    up = np.roll(spins, 1, axis=0)
    down = np.roll(spins, -1, axis=0)
    left = np.roll(spins, 1, axis=1)
    right = np.roll(spins, -1, axis=1)

    nearest_neighbors = up + down + left + right

    energy = -np.sum(spins * nearest_neighbors) / 2

    magnetization = np.abs(np.sum(spins))

    return energy, magnetization


def metropolis_step(temperature, size, spins, i, j):
    nearest_neighbors = (
        spins[(i + 1) % size, j]
        + spins[i, (j + 1) % size]
        + spins[(i - 1) % size, j]
        + spins[i, (j - 1) % size]
    )
    delta = 2 * spins[i, j] * nearest_neighbors

    if (delta <= 0) or (np.random.rand() <= np.exp(-delta / temperature)):
        spins[i, j] = -spins[i, j]

    return spins


def update_lattice(mc_steps, spins, temperature, size):
    for _ in range(mc_steps):
        for i, j in it.product(range(size), repeat=2):
            spins = metropolis_step(temperature, size, spins, i, j)

    return spins


def run(mc_steps, each, temperature, size):
    nviz = int(mc_steps / each)

    spins = initializate_lattice(size, ordered=False)
    thermodynamics = [statistics(spins)]

    for _ in range(nviz):
        spins = update_lattice(each, spins, temperature, size)
        thermodynamics.append(statistics(spins))

    energy, magnetization = np.hsplit(np.asarray(thermodynamics), 2)

    plt.scatter(list(range(0, mc_steps + 1, each)), magnetization / size ** 2)
    plt.show()


if __name__ == "__main__":
    mc_steps, each = 10000, 100

    critic_temperature = 2.269185
    temperature = 0.8 * critic_temperature

    size = 32

    run(mc_steps, each, temperature, size)
