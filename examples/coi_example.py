#!/usr/bin/env python

# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Combination of Extremum Seeking and Common Optimization Interfaces.

This example shows the same basic game as ``simple_example.py``. But
instead of running on a bare callback function, we package the controls
problem into a class that derives from the `COI`_. This allows us to
better compartmentalize the code and more cleanly separate the rendering
from the update logic.

.. _COI: https://cernml-coi.docs.cern.ch/
"""

from __future__ import annotations

import typing as t

import gym
import numpy as np
from matplotlib import pyplot

from cernml import coi, mpl_utils
from cernml.extremum_seeking import ExtremumSeeker

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


class HideAndSeekGame(coi.SingleOptimizable):
    """Game of hide and seek on a 2D plane.

    The game has a "goal" and a "seeker". The seeker performs Brownian
    motion on each time step. Furthermore, it picks a new position at
    random every 100 time steps.

    The seeker's goal is to get close to the goal and track it as
    closely as possible. On each time step, it may pick an arbitrary
    position for itself. The feedback is the distance between it and the
    goal.

    The renderer shows both the game board and the progress. The game
    board shows both the goal (orange) and the seeker (blue) as circles.
    The progress is a graph of goal–seeker distance over time. We expect
    it to approach zero and stay there.
    """

    metadata = {
        "render.modes": ["human", "matplotlib_figures"],
        "cern.machine": coi.Machine.NO_MACHINE,
        "cern.japc": False,
    }

    param_names = ["X", "Y"]
    optimization_space = gym.spaces.Box(-5.0, 5.0, shape=(2,))
    objective_range = (0.0, float(np.linalg.norm([10.0, 10.0])))
    objective_name = "Distance"

    def __init__(self) -> None:
        self.seeker = np.zeros(self.optimization_space.shape)
        self.goal = np.zeros(self.optimization_space.shape)
        self.history_indices: t.List[int] = []
        self.history_costs: t.List[float] = []
        self.renderer = mpl_utils.FigureRenderer.from_callback(self._iter_updates)

    def get_initial_params(self) -> np.ndarray:
        self.seeker = self.optimization_space.sample()
        self.goal = self.optimization_space.sample()
        return np.copy(self.seeker)

    def compute_single_objective(self, params: np.ndarray) -> float:
        if not (len(self.history_costs) + 1) % 100:
            self.goal = self.optimization_space.sample()
        self.seeker = np.copy(params)
        self.goal = np.clip(
            self.goal + np.random.normal(scale=0.05, size=self.goal.shape),
            self.optimization_space.low,
            self.optimization_space.high,
        )
        cost = float(np.linalg.norm(self.seeker - self.goal))
        self.history_indices.append(len(self.history_indices))
        self.history_costs.append(cost)
        return cost

    def _iter_updates(self, figure: Figure) -> mpl_utils.RenderGenerator:
        # Create two axes for board and history.
        ax_board: Axes
        ax_history: Axes
        [ax_board, ax_history] = figure.subplots(nrows=2, squeeze=False)
        # The board has a fixed size. Put two markers on it.
        ax_board.set_xlim(
            self.optimization_space.low[0], self.optimization_space.high[0]
        )
        ax_board.set_ylim(
            self.optimization_space.low[1], self.optimization_space.high[1]
        )
        line_seeker: Line2D
        line_goal: Line2D
        [line_seeker, line_goal] = ax_board.plot(*self.seeker, "o", *self.goal, "o")
        ax_board.grid()
        ax_board.legend(["Seeker", "Goal"])
        ax_board.set_title("Board")
        # The history grows over time. Put an empty line in it for now.
        line_history: Line2D
        [line_history] = ax_history.plot(self.history_indices, self.history_costs)
        ax_history.set_xlabel("Step")
        ax_history.set_ylabel("Distance")
        ax_history.set_title("Progress")
        ax_history.set_ylim(*self.objective_range)
        figure.tight_layout()
        while True:
            # Done drawing. Yield back to the caller.
            yield
            # Update marker positions and history.
            line_seeker.set_data(*self.seeker)
            line_goal.set_data(*self.goal)
            line_history.set_data(self.history_indices, self.history_costs)
            # Recalculate the bounding box.
            ax_history.relim()
            # Apply the new bounding box.
            ax_history.autoscale_view(scaley=False)

    def render(self, mode: str = "human") -> t.Optional[mpl_utils.MatplotlibFigures]:
        if mode in ["human", "matplotlib_figures"]:
            return self.renderer.update(mode)
        return super().render(mode)


def main() -> None:
    """Main function."""
    done = False

    def on_close(_evt: t.Any) -> None:
        nonlocal done
        done = True

    # Initialize the game and render it once so we have a figure.
    game = HideAndSeekGame()
    init = game.get_initial_params()
    game.render()
    figure = game.renderer.figure
    assert figure is not None
    # Now we can hook into the figure to stop the program when the
    # window is closed.
    figure.canvas.mpl_connect("close_event", on_close)

    # Create the optimizer. The parameters have been selected so that it
    # tracks the goal well without oscillating too much around it.
    seeker = ExtremumSeeker(oscillation_size=0.5, gain=5.0)
    # Turn it into a generator so we don't have to call
    # `calc_next_step()` manuallt.
    generator = seeker.make_generator(
        x0=init,
        bounds=(game.optimization_space.low, game.optimization_space.high),
    )
    params = next(generator).params
    while not done:
        # Give the GUI some time to draw.
        pyplot.pause(0.1)
        # The main loop.
        cost = game.compute_single_objective(params)
        params = generator.send(cost).params
        game.render()


if __name__ == "__main__":
    main()
