#!/usr/bin/env python

"""Play a game of hide and seek using Extremum Seeking (ES).

The game is played in 2D. It has a goal (orange) and a seeker (blue).
The goal moves randomly in small steps. In addition, its position on the
game board is completely randomized every 100 time steps.

The seeker is controlled by the ES algorithm. Its objective is to track
the goal as closely as possible. Its only feedback is the current
distance from the goal. This is a convex function, so ES is guaranteed
to solve the problem.
"""

from __future__ import annotations

import typing as t

import numpy as np
from matplotlib import pyplot

from cernml.extremum_seeking import ExtremumSeeker

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


def main() -> None:
    """Main function."""
    # Create a callback to break out game loop as soon as the window is
    # closed.
    done = False

    def on_close(_evt: t.Any) -> None:
        nonlocal done
        done = True

    pyplot.gcf().canvas.mpl_connect("close_event", on_close)

    # Set up bounds, optimizer and the two players. We turn the
    # ExtremumSeeker into a generator because we want to control the
    # game loop, but we also don't want to call `calc_next_step()`
    # manually.
    lower, upper = np.array([[-5.0, -5.0], [5.0, 5.0]])
    generator = ExtremumSeeker(oscillation_size=0.5, gain=5.0).make_generator(
        np.random.normal(scale=3.0, size=2), bounds=(lower, upper)
    )
    seeker = next(generator).params
    goal = np.random.normal(scale=3.0, size=2)

    # Set up the game board plot.
    pyplot.subplot(211)
    pyplot.xlim(lower[0], upper[0])
    pyplot.ylim(lower[1], upper[1])
    line_seeker: Line2D
    line_goal: Line2D
    [line_seeker, line_goal] = pyplot.plot(*seeker, "o", *goal, "o")
    pyplot.legend(["Seeker", "Goal"])
    pyplot.grid()
    pyplot.title("Board")

    # Set up the history plot.
    ax_history: Axes = pyplot.subplot(212)
    history_indices: t.List[int] = []
    history_costs: t.List[float] = []
    line_history: Line2D
    [line_history] = pyplot.plot(history_indices, history_costs)
    pyplot.xlabel("Step")
    pyplot.ylabel("Distance")
    pyplot.title("Progress")
    pyplot.ylim(0.0, np.linalg.norm([10.0, 10.0]))
    pyplot.tight_layout()

    # Main loop!
    while not done:
        # Give the GUI some time to process events.
        pyplot.pause(0.1)
        # Periodically rerandomize the goal, otherwise do Brownian
        # motion within bounds.
        if not (len(history_indices) + 1) % 100:
            goal = np.random.uniform(lower, upper)
        else:
            goal = np.clip(
                goal + np.random.normal(scale=0.05, size=goal.shape), lower, upper
            )
        # Update the seeker position.
        cost = np.linalg.norm(seeker - goal)
        seeker = generator.send(cost).params
        # Record the latest cost function value.
        history_indices.append(len(history_indices))
        history_costs.append(cost)
        # Update all plots.
        line_seeker.set_data(*seeker)
        line_goal.set_data(*goal)
        line_history.set_data(history_indices, history_costs)
        # Recalculate the history bounding box, since the graph grows.
        ax_history.relim()
        # And apply the new bounding box.
        ax_history.autoscale_view(scaley=False)


if __name__ == "__main__":
    main()
