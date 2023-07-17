import itertools as it

import matplotlib.pyplot as plt

import numpy as np


class IsingModel:
    def __init__(
        self,
        mc_steps=10000,
        each=100,
        temperature=2.269185,
        size=32,
        ordered=True,
        visualize=True,
    ):
        self.mc_steps = mc_steps
        self.each = each
        self.temperature = temperature
        self.size = size
        self.ordered = ordered
        self.visualize = visualize

    def _initializate_lattice(self):
        self.spins_ = (
            np.ones((self.size, self.size))
            if self.ordered
            else np.random.choice([1, -1], size=(self.size, self.size))
        )

    def _statistics(self):
        up = np.roll(self.spins_, 1, axis=0)
        down = np.roll(self.spins_, -1, axis=0)
        left = np.roll(self.spins_, 1, axis=1)
        right = np.roll(self.spins_, -1, axis=1)

        nearest_neighbors = up + down + left + right

        energy = -np.sum(self.spins_ * nearest_neighbors) / 2

        magnetization = np.abs(np.sum(self.spins_))

        return energy, magnetization

    def _metropolis_step(self, i, j):
        nearest_neighbors = (
            self.spins_[(i + 1) % self.size, j]
            + self.spins_[i, (j + 1) % self.size]
            + self.spins_[(i - 1) % self.size, j]
            + self.spins_[i, (j - 1) % self.size]
        )
        delta = 2 * self.spins_[i, j] * nearest_neighbors

        if (delta <= 0) or (
            np.random.rand() <= np.exp(-delta / self.temperature)
        ):
            self.spins_[i, j] = -self.spins_[i, j]

    def _update_lattice(self):
        for _ in range(self.each):
            for i, j in it.product(range(self.size), repeat=2):
                self._metropolis_step(i, j)

    def run(self):
        self.nviz_ = int(self.mc_steps / self.each)

        self._initializate_lattice()
        self.thermodynamics_ = [self._statistics()]

        for _ in range(self.nviz_):
            self._update_lattice()
            self.thermodynamics_.append(self._statistics())

        energy, magnetization = np.hsplit(np.asarray(self.thermodynamics_), 2)

        if self.visualize:
            plt.scatter(
                list(range(0, self.mc_steps + 1, self.each)),
                magnetization / self.size ** 2,
            )
            plt.show()


def main():
    ising = IsingModel(temperature=1.8, ordered=False)
    ising.run()


if __name__ == "__main__":
    main()
