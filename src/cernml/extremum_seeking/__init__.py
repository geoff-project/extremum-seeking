# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
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

The extremum seeking algorithm provides both an interface for numeric
optimization (locating an optimum) and for adaptive control (tracking a
drifting/noisy optimum). It also provides a coroutine-based interface,
`~ExtremumSeeker.make_generator()`, to leave the control loop in the
caller's hand.

Small examples
--------------

Defining a cost function and creating an `ExtremumSeeker` object:

    >>> rng = np.random.default_rng(0)
    >>> loc = np.zeros(2)
    >>> def cost_function(params: np.ndarray) -> float:
    ...     drift = rng.normal(scale=1e-2, size=loc.shape)
    ...     noise = rng.normal(scale=1e-3, size=loc.shape)
    ...     loc[:] += drift
    ...     cost = np.linalg.norm(loc + noise - params)
    ...     return cost
    >>> seeker = ExtremumSeeker(oscillation_size=0.1)

Executing a single control step:

    >>> x0 = rng.normal(0.1, size=loc.shape)
    >>> seeker.calc_next_step(x0, cost=cost_function(x0), step=0)
    array([0.26159863, 0.03066484])

Creating a generator that receives cost values and yields the next
parameter to evaluate:

    >>> gen = seeker.make_generator(x0)
    >>> cost = None
    >>> for i in range(10):
    ...     it = gen.send(cost)
    ...     cost = cost_function(it.params)
    >>> it.params
    array([ 0.16964995, -0.09272651])

Running an optimization loop:

    >>> res = seeker.optimize(cost_function, x0, max_calls=10)
    >>> print(res)
    params: [ 0.16998328 -0.09349066]
      cost: 0.18957208
       nit: 10


Running an optimization loop until the cost is sufficiently small:

    >>> res = seeker.optimize(cost_function, x0, cost_goal=0.01)
    >>> round(cost_function(res.params), 6)
    0.010504

Passing a callback function to the optimization loop:

    >>> def printer(seeker: ExtremumSeeker, iteration: Iteration):
    ...     print("Cost:", round(iteration.cost, 6))
    >>> _ = seeker.optimize(cost_function, x0, max_calls=1, callbacks=printer)
    Cost: 0.621505

Passing multiple callbacks, one of which ends the loop immediately by
returning `True`:

    >>> def make_printer(text: str) -> Callback:
    ...     def callback(*args):
    ...         print(text)
    ...         terminate = text == "foo"
    ...         return terminate
    ...     return callback
    >>> _ = seeker.optimize(
    ...     cost_function,
    ...     x0,
    ...     callbacks=[make_printer("foo"), make_printer("bar")],
    ... )
    foo
    bar

The :doc:`/examples/index` page contains more comprehensive example
programs.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np

Bounds = t.Tuple[np.ndarray, np.ndarray]
"""Lower and upper bounds for the search space.

Bounds are specified as a tuple :samp:`({lower}, {upper})` with the same
shape as the parameters.
"""

Callback = t.Callable[["ExtremumSeeker", "Iteration"], t.Optional[bool]]
"""Signature of callbacks for `optimize`.

Each callback is called at the end of each iteration with 2 arguments:
the `ExtremumSeeker` instance and the current `Iteration`.

If any callback returns any truth-like value, optimization terminates.
If the return value is false-like (including `None`), optimization
continues. This is so that a callback without return value never
terminates the optimization.
"""


def optimize(
    func: t.Callable[[np.ndarray], float],
    x0: np.ndarray,  # pylint: disable=invalid-name
    *,
    max_calls: t.Optional[int] = None,
    cost_goal: t.Optional[float] = None,
    callbacks: t.Union[Callback, t.Iterable[Callback]] = (),
    bounds: t.Optional[Bounds] = None,
    gain: float = 0.2,
    oscillation_size: float = 0.1,
    oscillation_sampling: int = 10,
    decay_rate: float = 1.0,
) -> OptimizeResult:
    """Run an optimization loop using ES.

    Args:
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
            within which the updated parameters should lie.
        gain: Scaling factor that is applied to the cost function. If
            positive (the default), the controller minimizes the cost
            function; if negative, the controller maximizes it.
        oscillation_size: Amplitude of the dithering oscillations if the
            cost were held constant.
        oscillation_sampling: Number of sampling points per dithering
            oscillation period. Larger values mean smaller time steps.
        decay_rate: An optional factor between 0 and 1 that reduces the
            *oscillation_size* after each step. Only used by
            `ExtremumSeeker.make_generator()` and
            `ExtremumSeeker.optimize()`.

    Returns:
        If *max_calls* has been supplied, this returns the final set of
        parameters. Otherwise, this method never returns.

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
    """The return value of `optimize()`.

    Args:
        params: The final set of parameters.
        cost: Corresponding value of the cost function. If the cost
            function was never evaluated or immediately raised an
            exception, this value is `~numpy.nan`.
        nit: The number of cost function evaluations.
    """

    params: np.ndarray
    cost: float = np.nan
    nit: int = 0

    def __str__(self) -> str:
        # Wrap cost and nit in NumPy scalars so that they obey
        # `np.printoptions`:
        fields = vars(self)
        width = max(map(len, fields))
        return "\n".join(
            f"{name:>{width}}: {_PrintOptionsAdapter(val)!s}"
            for name, val in fields.items()
        )


@dataclass
class Iteration:
    """Iteration-specific data for the ES algorithm.

    Args:
        nit: The number of cost function evaluations so far. This is
            always at least 1.
        params: The current set of parameters.
        cost: The cost associated with *params*.
        amplitude: An additional scaling factor applied to
            the *oscillation_size* of `ExtremumSeeker`. This is used to
            decay or otherwise dynamically adjust the dithering
            amplitudes.
        bounds: An optional set of bounds on the parameters. If
            supplied, they will be clipped to this space.
    """

    params: np.ndarray
    cost: float = np.nan
    nit: int = 0
    amplitude: float = 1.0
    bounds: t.Optional[Bounds] = None


class ExtremumSeeker:

    """Extremum-seeking controller. See module docstring for more info.

    Args:
        gain: Scaling factor that is applied to the cost function. If
            positive (the default), the controller minimizes the cost
            function; if negative, the controller maximizes it.
        oscillation_size: Amplitude of the dithering oscillations if the
            cost were held constant.
        oscillation_sampling: Number of sampling points per dithering
            oscillation period. Larger values mean smaller time steps.
        decay_rate: An optional factor between 0 and 1 that reduces the
            *oscillation_size* after each step. Only used by
            `make_generator()` and `optimize()`.

    Attributes:
        gain: The cost function scaling factor.
        oscillation_size: The dithering oscillation amplitude.
        oscillation_sampling: The number of samples per period.
        decay_rate: The amplitude decay rate.

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

    def get_dithering_freqs(self, ndim: int) -> np.ndarray:
        """Calculate the frequencies necessary for dithering.

        This returns a 1D array of length *ndim*, one for each parameter
        that is being controlled. They are chosen to prevent resonance
        between different parameters.
        """
        return np.linspace(self._W_MIN, self._W_MAX, ndim)

    def calc_next_step(
        self,
        params: np.ndarray,
        *,
        cost: float,
        step: int,
        amplitude: float = 1.0,
        bounds: t.Optional[Bounds] = None,
    ) -> np.ndarray:
        """Perform one step of the ES algorithm.

        Args:
            params: The previous set of parameters.
            cost: The cost associated with the previous parameters.
            step: The index of the current step.
            amplitude: Scaling factor on *oscillation_size*; can
                be used to implement amplitude decay.
            bounds: If passed, a tuple :samp:`({lower}, {upper})` of
                bounds within which the updated parameters should lie.

        Returns:
            The next set of parameters. An array with the same shape as
            *params*.

        Example:

            >>> seeker = ExtremumSeeker()
            >>> seeker.calc_next_step(np.zeros(2), cost=0.0, step=0)
            array([0.03590392, 0.06283185])
        """
        iteration = Iteration(params, cost, step, amplitude, bounds)
        return _calc_next_step(self, iteration)

    def make_generator(
        self,
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        bounds: t.Optional[Bounds] = None,
    ) -> t.Generator[Iteration, float, None]:
        """Create a generator of parameter suggestions.

        Args:
            x0: The initial set of parameters to suggest.
            bounds: If passed, a tuple :samp:`({lower}, {upper})` of
                bounds within which the updated parameters should lie.

        Returns:
            A generator *gen* that yields arrays, each of them being a
            result of `calc_next_step()`. The next cost function value
            is passed in via :samp:`gen.send({cost})`. After *max_calls*
            steps, the generator raises `StopIteration` with the final
            set of parameters as value.

        Example:

            >>> def cost_function(x):
            ...     return np.mean(x*x)
            >>> seeker = ExtremumSeeker()
            >>> gen = seeker.make_generator(np.zeros(2))
            >>> it = next(gen)
            >>> cost = cost_function(it.params)
            >>> for _ in range(10):
            ...     it = gen.send(cost)
            ...     cost = cost_function(it.params)
            >>> it.params
            array([-0.07720379,  0.00018237])
        """
        iteration = Iteration(np.copy(x0), bounds=bounds)
        while True:
            iteration.nit += 1
            iteration.cost = yield iteration
            if iteration.cost is None:
                raise TypeError("no cost passed; make sure to call `self.send(cost)`")
            if np.isnan(iteration.cost):
                raise ValueError(
                    f"cost is NaN (not a number) after {iteration.nit} ES step(s)"
                )
            iteration.params = _calc_next_step(self, iteration)
            iteration.amplitude *= self.decay_rate

    def optimize(
        self,
        func: t.Callable[[np.ndarray], float],
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        max_calls: t.Optional[int] = None,
        cost_goal: t.Optional[float] = None,
        callbacks: t.Union[Callback, t.Iterable[Callback]] = (),
        bounds: t.Optional[Bounds] = None,
    ) -> OptimizeResult:
        """Run an optimization loop using ES.

        Args:
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
                bounds within which the updated parameters should lie.

        Returns:
            If *max_calls* has been supplied, this returns the final set
            of parameters. Otherwise, this method never returns.

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
            return OptimizeResult(x0)
        callbacks = _consolidate_callbacks(callbacks, max_calls, cost_goal)
        generator = self.make_generator(x0, bounds=bounds)
        iteration = next(generator)
        iteration.cost = func(iteration.params)
        while not callbacks(self, iteration):
            iteration = generator.send(iteration.cost)
            iteration.cost = func(iteration.params)
        return _make_result_from_iteration(iteration)


def _consolidate_callbacks(
    callbacks: t.Union[Callback, t.Iterable[Callback]] = (),
    max_calls: t.Optional[int] = None,
    cost_goal: t.Optional[float] = None,
) -> _CallbackList:
    if isinstance(callbacks, t.Iterable):
        callbacks = _CallbackList(callbacks)
    else:
        callbacks = _CallbackList([callbacks])
    if cost_goal is not None:
        callbacks.append(_make_cost_goal_callback(cost_goal))
    if max_calls is not None:
        callbacks.append(_make_max_calls_callback(max_calls))
    return callbacks


class _CallbackList(t.List[Callback]):
    """List of callbacks.

    Calling this list always calls all elements. It ends optimization if
    any of the elements does.
    """

    def __call__(self, seeker: ExtremumSeeker, iteration: Iteration) -> bool:
        # Collect bools into a list first to ensure that each callback
        # is called. `any()` stops on the first True it finds.
        # pylint: disable = use-a-generator
        return any([bool(cb(seeker, iteration)) for cb in self])


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


def _calc_next_step(seeker: ExtremumSeeker, data: Iteration) -> np.ndarray:
    """Perform one step of the ES algorithm."""
    # Ensure that we have a flat array.
    params = np.asanyarray(data.params)
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


def _check_bounds_shape(ndim: int, lower: np.ndarray, upper: np.ndarray) -> None:
    if np.shape(lower) != (ndim,):
        raise ValueError(
            f"lower bound has wrong shape: expected ({ndim},), "
            f"found {np.shape(lower)}"
        )
    if np.shape(upper) != (ndim,):
        raise ValueError(
            f"upper bound has wrong shape: expected ({ndim},), "
            f"found {np.shape(upper)}"
        )


class _PrintOptionsAdapter:
    """Make scalars obey `np.printoptions()`."""

    # pylint: disable = too-few-public-methods

    def __init__(self, value: t.Any) -> None:
        self._value = value

    def __str__(self) -> str:
        if np.isscalar(self._value):
            return str(np.array([self._value]))[1:-1]
        return str(self._value)
