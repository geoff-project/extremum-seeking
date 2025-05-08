# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# pylint: disable = missing-function-docstring
# pylint: disable = missing-class-docstring
# pylint: disable = missing-module-docstring

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from cernml import extremum_seeking as es


@pytest.mark.parametrize("max_calls", [0, 1, 10])
def test_max_calls_reached(max_calls: int) -> None:
    cost_function = Mock(return_value=0.0)
    es.optimize(cost_function, np.zeros(2), max_calls=max_calls)
    assert cost_function.call_count == max_calls


def test_params_and_cost_in_sync() -> None:
    def cost_function(params: np.ndarray) -> float:
        return np.mean(np.square(params))

    def callback(_seeker: es.ExtremumSeeker, iteration: es.Iteration) -> None:
        assert cost_function(iteration.params) == iteration.cost

    res = es.optimize(cost_function, np.zeros(2), max_calls=20, callbacks=callback)
    assert cost_function(res.params) == res.cost


def test_decay_rate_reduces_amplitude() -> None:
    gen = es.ExtremumSeeker(decay_rate=0.5).make_generator(np.zeros(2))
    iteration = next(gen)
    for expected in [1.0, 0.5, 0.25, 0.125]:
        assert iteration.amplitude == expected
        iteration = gen.send(0.0)


def test_custom_amplitude_passed_through() -> None:
    expected = (2**-i for i in range(10))

    def callback(_seeker: es.ExtremumSeeker, iteration: es.Iteration) -> None:
        assert iteration.amplitude == next(expected)
        iteration.amplitude *= 0.5

    res = es.optimize(
        Mock(return_value=0.0), np.zeros(2), max_calls=10, callbacks=callback
    )
    assert np.allclose(res.params, np.array([0.02500405, 0.02193172]))


@pytest.mark.parametrize("gain", [0.0, np.inf, np.nan])
def test_raises_on_bad_gain(gain: float) -> None:
    with pytest.raises(ValueError, match="gain must not be zero:"):
        es.ExtremumSeeker(gain=gain)


@pytest.mark.parametrize("decay_rate", [-1.0, 0.0, 2.0, np.inf, np.nan])
def test_raises_on_bad_decay_rate(decay_rate: float) -> None:
    with pytest.raises(ValueError, match="decay_rate must be between 0 and 1:"):
        es.ExtremumSeeker(decay_rate=decay_rate)


def test_bounds_clip() -> None:
    bounds = 0.1 * np.ones(2)
    res = es.optimize(
        lambda x: np.mean(x * x),
        x0=np.zeros(2),
        max_calls=2,
        oscillation_size=1.0,
        bounds=(-bounds, bounds),
    )
    assert np.array_equal(res.params, bounds)


@pytest.mark.parametrize(
    ("bad_bound_name", "lower_shape", "upper_shape"),
    [("lower", 10, 2), ("upper", 2, 10)],
)
def test_bad_bounds(bad_bound_name: str, lower_shape: int, upper_shape: int) -> None:
    bounds = (-np.ones(lower_shape), np.ones(upper_shape))
    with pytest.raises(ValueError, match=bad_bound_name):
        es.optimize(Mock(return_value=0.0), x0=np.zeros(2), bounds=bounds)


def test_cost_is_none() -> None:
    gen = es.ExtremumSeeker().make_generator(np.zeros(2))
    next(gen)
    with pytest.raises(TypeError, match=r"send\(cost\)"):
        next(gen)


def test_cost_is_nan() -> None:
    cost_function = Mock(return_value=np.nan)
    with pytest.raises(ValueError, match="NaN"):
        es.optimize(cost_function, np.zeros(2))


def test_cost_goal_stops_optimization() -> None:
    cost_goal = 0.001
    cost_function = Mock(side_effect=lambda x: np.mean(np.square(x)))
    res = es.optimize(cost_function, x0=0.2 * np.ones(2), cost_goal=cost_goal)
    assert cost_function(res.params) == res.cost
    assert res.cost < cost_goal
    assert cost_function.call_count == 749


@pytest.mark.parametrize("max_calls", [0, 1, 10])
def test_callback_stops_optimization(max_calls: int) -> None:
    cost_function = Mock(return_value=0.0)
    callback = Mock(return_value=False)
    es.optimize(cost_function, np.zeros(2), max_calls=max_calls, callbacks=callback)
    assert callback.call_count == max_calls


def test_always_call_all_callbacks() -> None:
    expected_calls = 8
    cost_function = Mock(return_value=0.0)
    callbacks = [
        Mock(
            name=f"Callback #{i_callback}",
            side_effect=[(i_callback == 1 and step == 7) for step in range(10)],
        )
        for i_callback in range(1, 4)
    ]
    es.optimize(cost_function, x0=np.zeros(2), callbacks=callbacks)
    assert cost_function.call_count == expected_calls
    for callback in callbacks:
        assert callback.call_count == expected_calls


def test_each_iteration_object_unique() -> None:
    iterations = []
    es.optimize(
        Mock(name="cost function", return_value=0.0),
        x0=np.zeros(2),
        callbacks=lambda _seeker, it: iterations.append(it),
        max_calls=2,
    )
    assert iterations[0] is not iterations[1]


def test_iteration_nit_value() -> None:
    iterations = []
    es.optimize(
        Mock(name="cost function", return_value=0.0),
        x0=np.zeros(2),
        callbacks=lambda _seeker, it: iterations.append(it),
        max_calls=3,
    )
    assert iterations[0].nit == 1
    assert iterations[1].nit == 2
    assert iterations[2].nit == 3
