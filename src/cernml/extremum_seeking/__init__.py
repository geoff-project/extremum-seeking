# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Implementation of extremum-seeking control.

This implementation follows the description by `Scheinker et al.`_. The
core idea is to let the parameters oscillate around a center point and
have the phase advance of the oscillation depend on the cost function.
ES spends more time where the cost function is low and less time where
it is high. This causes a slow drift in the parameter space towards
global minima.

.. _Scheinker et al.: https://doi.org/10.1002/acs.3097

This package provides both an interface for numeric optimization
(locating an optimum) and for adaptive control (tracking
a drifting/noisy optimum). It also provides a :term:`generator`-based
interface (`.make_generator()`) to leave the control loop in the
caller's hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    import sys
    from collections.abc import Generator, Iterable
    from typing import Any, Callable, SupportsFloat

    from numpy.typing import NDArray

    if sys.version_info < (3, 10):
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias

__all__ = (
    "Bounds",
    "Callback",
    "ExtremumSeeker",
    "Iteration",
    "OptimizeResult",
    "optimize",
)

Bounds: TypeAlias = "tuple[NDArray[np.floating], NDArray[np.floating]]"

Callback: TypeAlias = "Callable[[ExtremumSeeker, Iteration], bool | None]"


def optimize(
    func: Callable[[NDArray[np.floating]], SupportsFloat],
    x0: NDArray[np.floating],
    *,
    max_calls: int | None = None,
    cost_goal: float | None = None,
    callbacks: Callback | Iterable[Callback] = (),
    bounds: Bounds | None = None,
    gain: float = 0.2,
    oscillation_size: float = 0.1,
    oscillation_sampling: int = 10,
    decay_rate: float = 1.0,
) -> OptimizeResult:
    """Run an optimization loop using ES.

    Args:
        func: The objective function to minimize.
        x0: The initial set of parameters to suggest.
        max_calls: If passed, end the generator after this many
            steps.
        cost_goal: If passed, end optimization once this threshold is
            crossed. if *gain* is positive (the default), the cost must
            be less than *cost_goal* to end optimization. If *gain* is
            negative, the cost must be greater.
        callbacks: If passed, should be either a callback function or a
            list of them. They are called on each iteration with 3
            arguments: the `ExtremumSeeker` instance, the current set of
            parameters, and the cost associated with these parameters.
            If any callback returns a truth-like value, optimization
            terminates.
        bounds: If passed, a tuple :samp:`({lower}, {upper})` of bounds
            within which the updated parameters will be clipped.
        gain: Scaling factor that is applied to the cost function. If
            positive (the default), the controller minimizes the cost
            function; if negative, the controller maximizes it.
        oscillation_size: Amplitude of the dithering oscillations if the
            cost were held constant.
        oscillation_sampling: Number of sampling points per dithering
            oscillation period. Larger values mean smaller time steps.
        decay_rate: An optional factor between 0 and 1 that reduces the
            *oscillation_size* after each step.

    Returns:
        Only guaranteed to return if you pass *max_calls*. Otherwise it
        *may* return if the conditions of *callbacks* or *cost_goal* are
        satisfied.

        If you pass none of these, this method never returns.

        If this method returns, it returns an `OptimizeResult` with the
        final set of parameters, the associated cost, and the final
        number of calls to the cost function.

    Example:
        >>> def cost_function(x):
        ...     return np.mean(x*x)
        >>> res = optimize(cost_function, x0=np.zeros(2), max_calls=10)
        >>> print(res)
        params: [-0.04486473 -0.06264948]
          cost: 0.0029689
           nit: 10
    """
    seeker = ExtremumSeeker(
        decay_rate=decay_rate,
        oscillation_size=oscillation_size,
        oscillation_sampling=oscillation_sampling,
        gain=gain,
    )
    return seeker.optimize(
        func=func,
        x0=x0,
        max_calls=max_calls,
        cost_goal=cost_goal,
        callbacks=callbacks,
        bounds=bounds,
    )


@dataclass
class OptimizeResult:
    """The :doc:`dataclass <std:library/dataclasses>` returned by `optimize()`."""

    params: NDArray[np.double]
    """The final set of parameters."""

    cost: float = np.nan
    """Value of the cost function evaluated at `params`. If the cost
    function was never evaluated or immediately raised an exception,
    this value is `~numpy.nan`."""

    nit: int = 0
    """The final number of cost function evaluations."""

    def __str__(self) -> str:
        # Wrap cost and nit so that they obey `np.printoptions`:
        fields = vars(self)
        width = max(map(len, fields))
        return "\n".join(
            f"{name:>{width}}: {_PrintOptionsAdapter(val)!s}"
            for name, val in fields.items()
        )


# TODO: Consider adding `slots=True` once we only support Python 3.10+.
@dataclass
class Step:
    """An *ongoing* iteration of the ES algorithm.

    This is returned by `.calc_next_step()` and yielded by the
    :term:`generator` returned by `.make_generator()`. You typically use
    `params` to evaluate the cost function and pass the result back to
    the algorithm.

    Examples:
        >>> seeker = ExtremumSeeker()
        >>> x0 = np.zeros(2)
        >>> step = Step(x0, bounds=(x0 - 1, x0 + 1))
        >>> step
        Step(params=array([0., 0.]), bounds=(array([-1., -1.]), array([1., 1.])))
        >>> step = seeker.calc_next_step(step, cost=0.0)
        >>> # You are free to modify `step` between calls.
        >>> step.bounds = None
        >>> step.amplitude = 0.5
        >>> step
        Step(params=array([0.0336145 , 0.05083204]), nit=1, amplitude=0.5)
    """

    params: NDArray[np.double]
    """The current set of parameters."""

    nit: int = 0
    """The number of cost function evaluations so far. Zero if the cost
    function hasn't been evaluated yet."""

    amplitude: float = 1.0
    """An additional scaling factor applied to the *oscillation_size* of
    `ExtremumSeeker`. This is used to decay or otherwise dynamically
    adjust the dithering amplitudes."""

    bounds: Bounds | None = None
    """Optional bounds on the parameters. If not None, subsequent
    parameters will be clipped to this space."""

    def with_cost(self, cost: SupportsFloat) -> Iteration:
        """Complete the step by passing the *cost* associated with `params`."""
        return Iteration(
            params=self.params,
            cost=float(cost),
            nit=self.nit + 1,
            amplitude=self.amplitude,
            bounds=self.bounds,
        )

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        args = [f"params={self.params!r}"]
        if self.nit != 0:
            args.append(f"nit={self.nit!r}")
        if self.amplitude != 1.0:
            args.append(f"amplitude={self.amplitude!r}")
        if self.bounds is not None:
            args.append(f"bounds={self.bounds!r}")
        return f"{cls_name}({', '.join(args)})"


# TODO: Consider adding `slots=True` once we only support Python 3.10+.
@dataclass
class Iteration:
    """A *completed* iteration of the ES algorithm.

    This is one of the arguments that a `Callback` may receive. You may
    inspect or store it, or use it to modulate the `amplitude` or
    `bounds` of the ongoing algorithm. You may also pass it to
    `.calc_next_step()` to proceed to the next `Step` of the algorithm.
    """

    params: NDArray[np.double]
    """The current set of parameters."""

    cost: float
    """The cost associated with `params`."""

    nit: int
    """The number of cost function evaluations so far. Because this
    object always requires a `cost`, this is always at least 1."""

    amplitude: float
    """An additional scaling factor applied to the *oscillation_size* of
    `ExtremumSeeker`. This is used to decay or otherwise dynamically
    adjust the dithering amplitudes."""

    bounds: Bounds | None
    """Optional bounds on the parameters. If not None, subsequent
    parameters will be clipped to this space."""


class ExtremumSeeker:
    """Extremum-seeking controller.

    Once created, you can use the controller via `calc_next_step()`,
    `make_generator()` or `optimize()`.

    Args:
        gain: The scaling factor that is applied to the cost function.
            If positive (the default), the controller minimizes the cost
            function; if negative, the controller maximizes it.
        oscillation_size: The amplitude of the dithering oscillations if
            the cost were held constant.
        oscillation_sampling: The number of sampling points per dithering
            oscillation period. Larger values mean smaller time steps.
        decay_rate: An optional factor between 0 and 1 that reduces the
            *oscillation_size* after each step.

    Each of these arguments is also available as an attribute.

    Example:
        >>> def func(x):
        ...     return np.mean(x*x)
        >>> seeker = ExtremumSeeker(gain=2)
        >>> res = seeker.optimize(func, np.zeros(3), max_calls=10)
        >>> print(res)
        params: [-0.04615712 -0.12537423 -0.06140994]
          cost: 0.00720679
           nit: 10
    """

    _W_MIN: float = 1.0
    _W_MAX: float = 1.75

    def __init__(
        self,
        *,
        gain: float = 0.2,
        oscillation_size: float = 0.1,
        oscillation_sampling: int = 10,
        decay_rate: float = 1.0,
    ) -> None:
        if gain == 0.0 or not np.isfinite(gain):
            raise ValueError(f"gain must not be zero: {gain}")
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be between 0 and 1: {decay_rate}")
        self.gain = gain
        self.oscillation_size = oscillation_size
        self.oscillation_sampling = oscillation_sampling
        self.decay_rate = decay_rate

    def get_time_step(self) -> float:
        """Calculate the ES time step size.

        Note that this time scale need not be related to the time scale
        of the system. This step size merely determines the oscillation
        of the ES algorithm.
        """
        return 2 * np.pi / (self.oscillation_sampling * self._W_MAX)

    def get_dithering_freqs(self, ndim: int) -> NDArray[np.double]:
        """Calculate the frequencies necessary for dithering.

        This returns a 1D array of length *ndim*, one for each parameter
        that is being controlled. They are chosen to prevent resonance
        between different parameters.
        """
        return np.linspace(self._W_MIN, self._W_MAX, ndim)

    @overload
    def calc_next_step(
        self, x0: NDArray[np.floating], /, *, cost: float, bounds: Bounds | None = None
    ) -> Step: ...
    @overload
    def calc_next_step(self, step: Step, /, *, cost: float) -> Step: ...
    @overload
    def calc_next_step(self, iteration: Iteration, /) -> Step: ...
    def calc_next_step(
        self,
        prev: Step | Iteration | NDArray[np.floating],
        /,
        *,
        cost: SupportsFloat | None = None,
        bounds: Bounds | None = None,
    ) -> Step:
        """Perform one step of the ES algorithm.

        For the first iteration, you can call the overload with *x0*,
        *cost* and *bounds*.

        For subsequent iterations, you should pass the previous `Step`
        and the *cost* associated with it---either as separate arguments
        or combined via `Step.with_cost()`.

        Args:
            x0: The initial set of parameters.
            step: The results of the previous iteration.
            cost: The cost associated with *x0* or the previous *step*.
            iteration: The previous *step* and its *cost* combined in
                a single object.
            bounds: If passed, a tuple :samp:`({lower}, {upper})` of
                bounds within which the updated parameters will be
                clipped. Must only be passed if *x0* is passed.

        Returns:
            The next set `Step` of the algorithm. Contains a new set of
            parameters, an incremented `~Step.nit` and a possibly
            decayed `~Step.amplitude`.

        Example:
            >>> seeker = ExtremumSeeker()
            >>> step = seeker.calc_next_step(np.zeros(2), cost=0.0)
            >>> step
            Step(params=array([0.0336145 , 0.05083204]), nit=1)
            >>> step = seeker.calc_next_step(step, cost=0.0)
            >>> step
            Step(params=array([0.06065271, 0.07024815]), nit=2)
        """
        if isinstance(prev, Iteration):
            if cost is not None:
                raise TypeError("first argument is an `Iteration`, no 'cost' allowed")
            if bounds is not None:
                raise TypeError("first argument is an `Iteration`, no 'bounds' allowed")
            iteration = prev
        elif isinstance(prev, Step):
            if cost is None:
                raise TypeError("first argument is a `Step`, 'cost' is required")
            if bounds is not None:
                raise TypeError("first argument is a `Step`, no 'bounds' allowed")
            iteration = prev.with_cost(cost)
        else:
            if cost is None:
                raise TypeError(
                    f"first argument is a `{type(prev).__name__}`, 'cost' is required"
                )
            prev = np.asarray(prev, dtype=np.double)
            iteration = Step(params=prev, bounds=bounds).with_cost(cost)
        if np.isnan(iteration.cost):
            raise ValueError(
                f"cost is NaN (not a number) after {iteration.nit} ES step(s)"
            )
        return Step(
            params=_calc_next_params(self, iteration),
            nit=iteration.nit,
            amplitude=iteration.amplitude * self.decay_rate,
            bounds=iteration.bounds,
        )

    def make_generator(
        self,
        x0: NDArray[np.floating],  # pylint: disable=invalid-name
        *,
        bounds: Bounds | None = None,
    ) -> Generator[Step, SupportsFloat, None]:
        r"""Create a generator of parameter suggestions.

        Args:
            x0: The initial set of parameters to suggest.
            bounds: If passed, a tuple :samp:`({lower}, {upper})` of
                bounds within which the updated parameters will be
                clipped.

        Returns:
            A :term:`generator` *gen* that yields `Step`\ s, each being
            a result of `calc_next_step()`. Pass in the next cost
            function value via :samp:`{gen}.send({cost})`.

        Example:
            >>> def cost_function(x):
            ...     return np.mean(x*x)
            ...
            >>> seeker = ExtremumSeeker()
            >>> gen = seeker.make_generator(np.zeros(2))
            >>> step = next(gen)
            >>> for _ in range(10):
            ...     cost = cost_function(step.params)
            ...     step = gen.send(cost)
            >>> step.params
            array([-0.07720379,  0.00018237])
        """
        step = Step(np.asarray(x0, dtype=np.double), bounds=bounds)
        while True:
            cost = yield step
            if cost is None:
                raise TypeError("no cost passed; make sure to call `self.send(cost)`")
            iteration = step.with_cost(cost)
            step = self.calc_next_step(iteration)

    def optimize(
        self,
        func: Callable[[NDArray[np.double]], SupportsFloat],
        x0: NDArray[np.floating],  # pylint: disable=invalid-name
        *,
        max_calls: int | None = None,
        cost_goal: float | None = None,
        callbacks: Callback | Iterable[Callback] = (),
        bounds: Bounds | None = None,
    ) -> OptimizeResult:
        """Run an optimization loop using ES.

        Args:
            func: The objective function to minimize.
            x0: The initial set of parameters to suggest.
            max_calls: If passed, end the generator after this many
                steps.
            cost_goal: If passed, end optimization once this threshold
                is crossed. if *gain* is positive (the default), the
                cost must be less than *cost_goal* to end optimization.
                If *gain* is negative, the cost must be greater.
            callbacks: If passed, should be either a callback function
                or a list of them. They are called on each iteration
                with 2 arguments: the `ExtremumSeeker` instance and the
                current `Iteration`. If any callback returns any
                truth-like value, optimization terminates.
            bounds: If passed, a tuple :samp:`({lower}, {upper})` of
                bounds within which the updated parameters will be
                clipped.

        Returns:
            Only guaranteed to return if you pass *max_calls*. Otherwise
            it *may* return if the conditions of *callbacks* or
            *cost_goal* are satisfied.

            If you pass none of these, this method never returns.

            If this method returns, it returns an `OptimizeResult` with
            the final set of parameters, the associated cost, and the
            final number of calls to the cost function.

        Example:
            >>> def cost_function(x):
            ...     return np.mean(x*x)
            >>> seeker = ExtremumSeeker()
            >>> res = seeker.optimize(
            ...     cost_function, x0=np.zeros(2), max_calls=10
            ... )
            >>> print(res)
            params: [-0.04486473 -0.06264948]
              cost: 0.0029689
               nit: 10
        """
        # Special case max_calls==0: Avoid calling any part of the
        # optimization loop, just return immediately.
        if max_calls is not None and max_calls <= 0:
            return OptimizeResult(np.asarray(x0, dtype=np.double))
        callbacks = _consolidate_callbacks(callbacks, max_calls, cost_goal)
        step = Step(x0, bounds=bounds)
        while True:
            iteration = step.with_cost(func(step.params))
            if callbacks(self, iteration):
                break
            step = self.calc_next_step(iteration)
        return _make_result_from_iteration(iteration)


def _consolidate_callbacks(
    callbacks: Callback | Iterable[Callback] = (),
    max_calls: int | None = None,
    cost_goal: float | None = None,
) -> _CallbackList:
    try:
        callbacks = _CallbackList(callbacks)  # type: ignore[arg-type]
    except TypeError:
        callbacks = _CallbackList([callbacks])  # type: ignore[list-item]
    if cost_goal is not None:
        callbacks.append(_make_cost_goal_callback(cost_goal))
    if max_calls is not None:
        callbacks.append(_make_max_calls_callback(max_calls))
    return callbacks


class _CallbackList(list[Callback]):
    """List of callbacks.

    Calling this list always calls all elements. It ends optimization if
    any of the elements does.
    """

    def __call__(self, seeker: ExtremumSeeker, iteration: Iteration) -> bool:
        # Collect bools into a list first to ensure that each callback
        # is called. `any()` stops on the first True it finds.
        # pylint: disable = use-a-generator
        return any([bool(cb(seeker, iteration)) for cb in self])  # noqa: C419


def _make_cost_goal_callback(cost_goal: float) -> Callback:
    def _cost_goal(seeker: ExtremumSeeker, iteration: Iteration) -> bool:
        cost = iteration.cost
        return (cost < cost_goal) if seeker.gain > 0.0 else (cost > cost_goal)

    return _cost_goal


def _make_max_calls_callback(max_calls: int) -> Callback:
    def _max_calls(_: ExtremumSeeker, iteration: Iteration) -> bool:
        return iteration.nit >= max_calls

    return _max_calls


def _make_result_from_iteration(iteration: Iteration) -> OptimizeResult:
    return OptimizeResult(iteration.params, iteration.cost, iteration.nit)


def _calc_next_params(seeker: ExtremumSeeker, data: Iteration) -> NDArray[np.double]:
    """Perform one step of the ES algorithm."""
    # Ensure that we have a flat array.
    params = np.asarray(data.params, dtype=np.double)
    [ndim] = params.shape
    time_step = seeker.get_time_step()
    # Choose frequency different for each dimension without
    # resonance between them.
    dithering_freqs = seeker.get_dithering_freqs(ndim)
    # Choose amplitudes such that integrating over all steps yields
    # an oscillation with amplitude `oscillation_size`.
    dithering_amplitudes = dithering_freqs * time_step * seeker.oscillation_size
    dithering_phases = dithering_freqs * time_step * data.nit + seeker.gain * data.cost
    # The actual calculation.
    next_step = dithering_amplitudes * np.cos(dithering_phases)
    next_params = params + data.amplitude * next_step
    # Ensure we remain within bounds.
    if data.bounds is not None:
        lower, upper = data.bounds
        _check_bounds_shape(ndim, lower, upper)
        next_params = np.clip(next_params, lower, upper)
    return next_params


def _check_bounds_shape(
    ndim: int, lower: NDArray[np.floating], upper: NDArray[np.floating]
) -> None:
    if np.shape(lower) != (ndim,):
        raise ValueError(
            f"lower bound has wrong shape: expected ({ndim},), found {np.shape(lower)}"
        )
    if np.shape(upper) != (ndim,):
        raise ValueError(
            f"upper bound has wrong shape: expected ({ndim},), found {np.shape(upper)}"
        )


class _PrintOptionsAdapter:
    """Make scalars obey `np.printoptions()`."""

    # pylint: disable = too-few-public-methods

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def __str__(self) -> str:
        if np.isscalar(self._value):
            return str(np.array([self._value]))[1:-1]
        return str(self._value)


del TYPE_CHECKING
