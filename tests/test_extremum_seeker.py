#!/usr/bin/env python

# pylint: disable = missing-function-docstring
# pylint: disable = missing-class-docstring
# pylint: disable = missing-module-docstring

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from cernml import extremum_seeking as es


@pytest.mark.parametrize("max_calls", [0, 1, 10])
def test_max_calls(max_calls: int) -> None:
    cost_function = Mock(return_value=0.0)
    es.optimize(cost_function, np.zeros(2), max_calls=max_calls)
    assert cost_function.call_count == max_calls


def test_params_and_cost_in_sync() -> None:
    def _cost_function(params: np.ndarray) -> float:
        return np.mean(np.square(params))

    def _callback(_seeker: es.ExtremumSeeker, iteration: es.Iteration) -> None:
        assert _cost_function(iteration.params) == iteration.cost

    res = es.optimize(_cost_function, np.zeros(2), max_calls=20, callbacks=_callback)
    assert _cost_function(res.params) == res.cost


@pytest.mark.parametrize("gain", [0.0, np.inf, np.nan])
def test_raises_on_bad_gain(gain: float) -> None:
    with pytest.raises(ValueError):
        es.ExtremumSeeker(gain=gain)


@pytest.mark.parametrize("decay_rate", [-1.0, 0.0, 2.0, np.inf, np.nan])
def test_raises_on_bad_decay_rate(decay_rate: float) -> None:
    with pytest.raises(ValueError):
        es.ExtremumSeeker(decay_rate=decay_rate)


def test_bounds() -> None:
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
    "bad_bound_name, lower_shape, upper_shape",
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


def test_cost_goal() -> None:
    cost_goal = 0.001
    cost_function = Mock(side_effect=lambda x: np.mean(np.square(x)))
    res = es.optimize(cost_function, x0=0.2 * np.ones(2), cost_goal=cost_goal)
    assert cost_function(res.params) == res.cost
    assert res.cost < cost_goal
    assert cost_function.call_count == 749


@pytest.mark.parametrize("max_calls", [0, 1, 10])
def test_callback(max_calls: int) -> None:
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
