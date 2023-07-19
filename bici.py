import itertools as it

import matplotlib.pyplot as plt

import numpy as np


class IsingModel:
    def __init__(
        self,
        mc_steps=10000,
        eq_steps=0,
        each=100,
        temperature=2.269185,
        size=32,
        ordered=True,
        visualize=True,
    ):
        self.mc_steps = mc_steps
        self.eq_steps = eq_steps
        self.each = each
        self.temperature = temperature
        self.size = size
        self.ordered = ordered
        self.visualize = visualize

        self.energy_, self.magnetization_ = [], []

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

        self.energy_.append(
            -np.sum(self.spins_ * nearest_neighbors) / 2 / self.spins_.size
        )
        self.magnetization_.append(np.abs(np.mean(self.spins_)))

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
        for i, j in it.product(range(self.size), repeat=2):
            self._metropolis_step(i, j)

    def run(self):
        self._initializate_lattice()

        for _ in range(self.eq_steps):
            self._update_lattice()

        self._statistics()

        if self.visualize:
            self.viz_ = IsingVisualization()
            self.viz_.show_frame(self)

        for i in range(self.eq_steps, self.mc_steps):
            self._update_lattice()

            if i % self.each == 0:
                self._statistics()

                if self.visualize:
                    self.viz_.show_frame(self)

        self.energy_ = np.asarray(self.energy_)
        self.magnetization_ = np.asarray(self.magnetization_)


class IsingVisualization:
    def __init__(self):
        self._fig, self._axd = plt.subplot_mosaic(
            [["left", "upper right"], ["left", "lower right"]],
            figsize=(10, 5),
            layout="constrained",
        )

    def show_frame(self, ising):
        self._axd["left"].clear()
        self._axd["left"].set_title(
            f"Ising model with {ising.spins_.size} spins at T={ising.temperature}"
        )
        self._axd["left"].set_xlabel("L")
        self._axd["left"].set_ylabel("L")
        self._axd["left"].imshow(ising.spins_, vmin=-1, vmax=1)

        self._axd["upper right"].clear()
        self._axd["upper right"].set_xlabel("MC step")
        self._axd["upper right"].set_ylabel(r"|M| / N$_{spins}$")
        self._axd["upper right"].scatter(
            list(range(0, ising.each * len(ising.magnetization_), ising.each)),
            ising.magnetization_,
        )

        self._axd["lower right"].clear()
        self._axd["lower right"].set_xlabel("MC step")
        self._axd["lower right"].set_ylabel(r"E / N$_{spins}$")
        self._axd["lower right"].scatter(
            list(range(0, ising.each * len(ising.energy_), ising.each)),
            ising.energy_,
        )

        plt.pause(0.001)


def main():
    ising = IsingModel(mc_steps=120, each=1, temperature=1.8, ordered=False)
    ising.run()


if __name__ == "__main__":
    main()
