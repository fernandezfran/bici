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

    def _show_viz_frame(self, axd):
        axd["left"].clear()
        axd["left"].set_title(
            f"Ising model with {self.spins_.size} spins at T={self.temperature}"
        )
        axd["left"].set_xlabel("L")
        axd["left"].set_ylabel("L")
        axd["left"].imshow(self.spins_)

        axd["upper right"].clear()
        axd["upper right"].set_xlabel("MC step")
        axd["upper right"].set_ylabel(r"|M| / N$_{spins}$")
        axd["upper right"].scatter(
            list(range(0, self.each * len(self.magnetization_), self.each)),
            self.magnetization_,
        )

        axd["lower right"].clear()
        axd["lower right"].set_xlabel("MC step")
        axd["lower right"].set_ylabel(r"E / N$_{spins}$")
        axd["lower right"].scatter(
            list(range(0, self.each * len(self.energy_), self.each)),
            self.energy_,
        )

        plt.tight_layout()
        plt.pause(0.001)

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

        ener, magn = self._statistics()
        self.energy_ = [ener / self.spins_.size]
        self.magnetization_ = [magn / self.spins_.size]

        if self.visualize:
            fig, axd = plt.subplot_mosaic(
                [["left", "upper right"], ["left", "lower right"]],
                figsize=(10, 5),
                layout="constrained",
            )
            self._show_viz_frame(axd)

        for _ in range(self.nviz_):
            self._update_lattice()

            ener, magn = self._statistics()
            self.energy_.append(ener / self.spins_.size)
            self.magnetization_.append(magn / self.spins_.size)

            if self.visualize:
                self._show_viz_frame(axd)

        self.energy_ = np.asarray(self.energy_)
        self.magnetization_ = np.asarray(self.magnetization_)


def main():
    ising = IsingModel(mc_steps=200, each=1, temperature=1.8, ordered=False)
    ising.run()


if __name__ == "__main__":
    main()
