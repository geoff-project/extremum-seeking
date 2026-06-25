# SPDX-FileCopyrightText: 2020-2026 CERN
# SPDX-FileCopyrightText: 2023-2026 GSI Helmholtzzentrum für Schwerionenforschung
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


def quadratic_cost_function(params: np.ndarray) -> float:
    return np.mean(np.square(params))


@pytest.mark.parametrize("max_calls", [0, 1, 10])
def test_max_calls_reached(max_calls: int) -> None:
    cost_function = Mock(name="cost function", return_value=0.0)
    es.optimize(cost_function, np.zeros(2), max_calls=max_calls)
    assert cost_function.call_count == max_calls


def test_callback_params_and_cost_in_sync() -> None:
    def callback(_seeker: es.ExtremumSeeker, iteration: es.Iteration) -> None:
        assert quadratic_cost_function(iteration.params) == iteration.cost

    res = es.optimize(
        quadratic_cost_function, np.zeros(2), max_calls=20, callbacks=callback
    )
    assert quadratic_cost_function(res.params) == res.cost


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
        Mock(name="cost function", return_value=0.0),
        np.zeros(2),
        max_calls=10,
        callbacks=callback,
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
        quadratic_cost_function,
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
        es.optimize(
            Mock(name="cost function", return_value=0.0), x0=np.zeros(2), bounds=bounds
        )


def test_cost_is_none() -> None:
    gen = es.ExtremumSeeker().make_generator(np.zeros(2))
    next(gen)
    with pytest.raises(TypeError, match=r"send\(cost\)"):
        next(gen)


def test_cost_is_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        es.optimize(Mock(name="cost function", return_value=np.nan), np.zeros(2))


def test_cost_goal_stops_optimization() -> None:
    cost_goal = 0.001
    cost_function = Mock(side_effect=quadratic_cost_function)
    res = es.optimize(cost_function, x0=0.2 * np.ones(2), cost_goal=cost_goal)
    assert cost_function(res.params) == res.cost
    assert res.cost < cost_goal
    assert cost_function.call_count == 749


@pytest.mark.parametrize("max_calls", [0, 1, 10])
def test_callback_stops_optimization(max_calls: int) -> None:
    cost_function = Mock(name="cost function", return_value=0.0)
    callback = Mock(name="callback", return_value=False)
    es.optimize(cost_function, np.zeros(2), max_calls=max_calls, callbacks=callback)
    assert callback.call_count == max_calls


def test_always_call_all_callbacks() -> None:
    expected_calls = 8
    cost_function = Mock(name="cost function", return_value=0.0)
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


def test_calc_next_step_bad_cost() -> None:
    seeker = es.ExtremumSeeker()
    with pytest.raises(TypeError, match="no 'cost' allowed$"):
        seeker.calc_next_step(Mock(es.Iteration), cost=0.0)


def test_calc_next_step_bad_bounds() -> None:
    seeker = es.ExtremumSeeker()
    with pytest.raises(TypeError, match="no 'bounds' allowed$"):
        seeker.calc_next_step(Mock(es.Iteration), bounds=Mock(name="bounds"))  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="no 'bounds' allowed$"):
        seeker.calc_next_step(Mock(es.Step), cost=0.0, bounds=Mock(name="bounds"))


def test_calc_next_step_no_cost() -> None:
    seeker = es.ExtremumSeeker()
    with pytest.raises(TypeError, match="'cost' is required$"):
        seeker.calc_next_step(Mock(es.Step))
    with pytest.raises(TypeError, match="'cost' is required$"):
        seeker.calc_next_step(np.zeros(3))  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Per-dimension oscillation_size
# ---------------------------------------------------------------------------


def test_oscillation_size_per_dim_scales_each_axis() -> None:
    """Per-axis oscillation_size scales each axis's first move proportionally."""
    # Use a constant zero cost so the only differences between runs come
    # from oscillation_size. Set a non-default amplitude on the seed step
    # so the test does not depend on Step's default amplitude.
    x0 = np.zeros(2)
    scalar_seeker = es.ExtremumSeeker(oscillation_size=1.0)
    scalar_step = scalar_seeker.calc_next_step(x0, cost=0.0)
    # Per-dim with axis-0 doubled and axis-1 left identical.
    vec_seeker = es.ExtremumSeeker(oscillation_size=np.array([2.0, 1.0]))
    vec_step = vec_seeker.calc_next_step(x0, cost=0.0)
    assert np.isclose(vec_step.params[0], 2.0 * scalar_step.params[0])
    assert np.isclose(vec_step.params[1], 1.0 * scalar_step.params[1])


def test_oscillation_size_wrong_shape_raises() -> None:
    seeker = es.ExtremumSeeker(oscillation_size=np.ones(3))
    with pytest.raises(ValueError, match="oscillation_size has wrong shape"):
        seeker.calc_next_step(np.zeros(2), cost=0.0)


def test_oscillation_size_nd_array_raises() -> None:
    seeker = es.ExtremumSeeker(oscillation_size=np.ones((2, 2)))
    with pytest.raises(ValueError, match="must be a scalar or a 1-D array"):
        seeker.calc_next_step(np.zeros(2), cost=0.0)


# ---------------------------------------------------------------------------
# Adaptive amplitude (AdaptiveAmplitude)
# ---------------------------------------------------------------------------


def test_adaptive_amplitude_large_error_near_max() -> None:
    schedule = es.AdaptiveAmplitude(
        cost_target=0.0,
        amplitude_min=0.1,
        amplitude_max=5.0,
        midpoint=0.3,
        sensitivity=7.0,
    )
    # error much larger than midpoint -> sigmoid saturates near 1 -> near max.
    assert schedule(10.0) == pytest.approx(5.0, abs=1e-6)


def test_adaptive_amplitude_small_error_near_min() -> None:
    schedule = es.AdaptiveAmplitude(
        cost_target=0.0,
        amplitude_min=0.1,
        amplitude_max=5.0,
        midpoint=10.0,  # midpoint deliberately far above zero
        sensitivity=7.0,  # so error=0 sits well below the 50% point
    )
    # error << midpoint -> sigmoid saturates near 0 -> near amplitude_min.
    assert schedule(0.0) == pytest.approx(0.1, abs=1e-6)


def test_adaptive_amplitude_at_midpoint_is_halfway() -> None:
    schedule = es.AdaptiveAmplitude(
        cost_target=0.0,
        amplitude_min=0.0,
        amplitude_max=10.0,
        midpoint=0.5,
        sensitivity=7.0,
    )
    # |cost - target| == midpoint -> sigmoid = 0.5 -> amp = (min + max)/2.
    assert schedule(0.5) == pytest.approx(5.0, abs=1e-6)


def test_adaptive_amplitude_drives_step_amplitude() -> None:
    """calc_next_step assigns the adaptive amplitude to the next Step."""
    schedule = es.AdaptiveAmplitude(
        cost_target=0.0,
        amplitude_min=0.1,
        amplitude_max=5.0,
        midpoint=0.3,
        sensitivity=7.0,
    )
    seeker = es.ExtremumSeeker(adaptive_amplitude=schedule)
    step = seeker.calc_next_step(np.zeros(2), cost=10.0)
    assert step.amplitude == pytest.approx(schedule(10.0))
    # And a low-error cost yields a small amplitude.
    step = seeker.calc_next_step(step, cost=0.0)
    assert step.amplitude == pytest.approx(schedule(0.0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cost_target": np.nan}, "cost_target must be finite"),
        ({"cost_target": 0.0, "amplitude_min": -1.0}, "amplitude_min"),
        (
            {"cost_target": 0.0, "amplitude_max": 0.0},
            "amplitude_max must be strictly positive",
        ),
        (
            {"cost_target": 0.0, "amplitude_min": 1.0, "amplitude_max": 0.5},
            "must be >=",
        ),
        ({"cost_target": 0.0, "midpoint": -1.0}, "midpoint"),
        ({"cost_target": 0.0, "sensitivity": 0.0}, "sensitivity"),
    ],
)
def test_adaptive_amplitude_bad_config(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        es.AdaptiveAmplitude(**kwargs)


def test_adaptive_amplitude_with_decay_rate_raises() -> None:
    schedule = es.AdaptiveAmplitude(cost_target=0.0)
    with pytest.raises(ValueError, match="non-default decay_rate"):
        es.ExtremumSeeker(adaptive_amplitude=schedule, decay_rate=0.5)


def test_adaptive_amplitude_disables_decay() -> None:
    """With an AdaptiveAmplitude schedule, decay_rate plays no role."""
    # amplitude_min == amplitude_max collapses the schedule to a constant.
    schedule = es.AdaptiveAmplitude(
        cost_target=0.0,
        amplitude_min=0.5,
        amplitude_max=0.5,
        midpoint=1.0,
        sensitivity=1.0,
    )
    seeker = es.ExtremumSeeker(adaptive_amplitude=schedule)
    step = seeker.calc_next_step(np.zeros(2), cost=2.0)
    assert step.amplitude == pytest.approx(0.5)
    # Even after many steps, amplitude stays at the adaptive value (no decay).
    for _ in range(10):
        step = seeker.calc_next_step(step, cost=2.0)
    assert step.amplitude == pytest.approx(0.5)
