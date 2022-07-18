"""Implementation of extremum-seeking control.

This implementation follows the description by [Scheinker et al][1]. The
core idea is to let the parameters oscillate around a center point and
have the phase advance of the oscillation depend on the cost function.
ES spends more time where the cost function is low (and phase advance
small) and less time where it is high. This causes a slow drift in the
parameter space towards global minima.

The extremum seeking algorithm provides both an interface for numeric
optimization (locating an optimum) and for adaptive control (tracking a
drifting/noisy optimum). It also provides a coroutine-based interface,
:meth:`~ExtremumSeeker.make_generator()`, to leave the control loop in
the caller's hand.

[1]: https://doi.org/10.1002/acs.3097
"""

from __future__ import annotations

import typing as t

import numpy as np

Callback = t.Callable[["ExtremumSeeker", np.ndarray, float], t.Optional[bool]]
"""Signature of callbacks for :func:`optimize`.

The arguments are the instance of the extremum seeker, the current
parameters and the cost associated with them. The callback should return
:obj:`None` or :obj:`False` if optimization should continue or
:obj:`True` if optimization should end.

This is so that a callback without return value never terminates the
optimization.
"""


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
            :meth:`make_generator()` and :meth:`optimize()`.

    Attributes:
        gain: The cost function scaling factor.
        oscillation_size: The dithering oscillation amplitude.
        oscillation_sampling: The number of samples per period.
        decay_rate: The amplitude decay rate.

    Example:

        >>> def func(x):
        ...     return np.mean(x*x)
        >>> seeker = ExtremumSeeker(gain=2)
        >>> seeker.optimize(func, np.zeros(3), max_calls=10)
        array([-0.01223014, -0.07974419, -0.00241286])
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
        if not gain:
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

    # This function defines one step of the ES algorithm at iteration i
    def calc_next_step(
        self,
        params: np.ndarray,
        *,
        cost: float,
        step: int,
        amplitude: float = 1.0,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Perform one step of the ES algorithm.

        Args:
            params: The previous set of parameters.
            cost: The cost associated with the previous parameters.
            step: The index of the current step.
            amplitude: Scaling factor on :attr:`oscillation_size`, can
                be used to implement amplitude decay.
            bounds: If passed, a tuple ``(lower, upper)`` of bounds
                within which the updated parameters should lie.

        Returns:
            The next set of parameters. An array with the same shape as
            *params*.

        Example:

            >>> seeker = ExtremumSeeker()
            >>> seeker.calc_next_step(np.zeros(2), cost=0.0, step=0)
            array([0.03590392, 0.06283185])
        """
        # Ensure that we have a flat array.
        params = np.asanyarray(params)
        [ndim] = params.shape
        time_step = self.get_time_step()
        # Choose frequency different for each dimension without
        # resonance between them.
        dithering_freqs = np.linspace(self._W_MIN, self._W_MAX, ndim)
        # Choose amplitudes such that integrating over all steps yields
        # an oscillation with amplitude `oscillation_size`.
        dithering_amplitudes = dithering_freqs * time_step * self.oscillation_size
        dithering_phases = dithering_freqs * time_step * step + self.gain * cost
        # The actual calculation.
        next_step = dithering_amplitudes * np.cos(dithering_phases)
        next_params = params + amplitude * next_step
        # Ensure we remain within bounds.
        if bounds is not None:
            lower, upper = bounds
            _check_bounds_shape(ndim, lower, upper)
            next_params = np.clip(next_params, lower, upper)
        return next_params

    def make_generator(
        self,
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
        max_calls: t.Optional[int] = None,
    ) -> t.Generator[np.ndarray, float, np.ndarray]:
        """Create a generator of parameter suggestions.

        Args:
            x0: The initial set of parameters to suggest.
            bounds: If passed, a tuple ``(lower, upper)`` of bounds
                within which the updated parameters should lie.
            max_calls: If passed, end the generator after this many
                steps.

        Returns:
            A generator *gen* that yields arrays, each of them being a
            result of :meth:`calc_next_step()`. The next cost function
            value is passed in via ``gen.send(cost)``. After *max_calls*
            steps, the generator raises :exc:`StopIteration` with the
            final set of parameters as value.

        Example:

            >>> def cost_function(x):
            ...     return np.mean(x*x)
            >>> seeker = ExtremumSeeker()
            >>> gen = seeker.make_generator(np.zeros(2), max_calls=10)
            >>> try:
            ...     params = next(gen)
            ...     while True:
            ...         cost = cost_function(params)
            ...         params = gen.send(cost)
            ... except StopIteration as exc:
            ...     params = exc.value
            >>> params
            array([-0.00914946, -0.00020783])
        """
        params = np.copy(x0)
        amplitude = 1.0
        i = 0
        while max_calls is None or i < max_calls:
            cost = yield params
            if cost is None:
                raise TypeError("no cost passed; make sure to call `self.send(cost)`")
            if np.isnan(cost):
                raise ValueError(f"cost is NaN (not a number) after {i} ES step(s)")
            params = self.calc_next_step(
                params=params,
                cost=cost,
                step=i,
                amplitude=amplitude,
                bounds=bounds,
            )
            amplitude *= self.decay_rate
            i += 1
        return params

    @t.overload
    def optimize(
        self,
        func: t.Callable[[np.ndarray], float],
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        max_calls: None = ...,
        cost_goal: t.Optional[float] = ...,
        callbacks: t.Union[Callback, t.Iterable[Callback]] = ...,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = ...,
    ) -> t.NoReturn:
        ...

    @t.overload
    def optimize(
        self,
        func: t.Callable[[np.ndarray], float],
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        max_calls: int,
        cost_goal: t.Optional[float] = ...,
        callbacks: t.Union[Callback, t.Iterable[Callback]] = ...,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = ...,
    ) -> np.ndarray:
        ...

    def optimize(
        self,
        func: t.Callable[[np.ndarray], float],
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        max_calls: t.Optional[int] = None,
        cost_goal: t.Optional[float] = None,
        callbacks: t.Union[Callback, t.Iterable[Callback]] = (),
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Run an optimization loop using ES.

        Args:
            x0: The initial set of parameters to suggest.
            max_calls: If passed, end the generator after this many
                steps.
            cost_goal: If passed, end optimization once this threshold
                is crossed. if :attr:`gain` is positive (the default),
                the cost must be less than *cost_goal* to end
                optimization. If :attr:`gain` is negative, the cost must
                be greater.
            callbacks: If passed, should be either a callback function
                or a list of them. They are called on each iteration
                with 3 arguments: the :class:`ExtremumSeeker` instance,
                the current set of parameters, and the cost associated
                with these parameters. If any callback returns a
                truth-like value, optimization terminates.
            bounds: If passed, a tuple ``(lower, upper)`` of bounds
                within which the updated parameters should lie.

        Returns:
            If *max_calls* has been supplied, this returns the final set
            of parameters. Otherwise, this method never returns.

        Example:

            >>> def cost_function(x):
            ...     return np.mean(x*x)
            >>> seeker = ExtremumSeeker()
            >>> seeker.optimize(
            ...     cost_function, x0=np.zeros(2), max_calls=10
            ... )
            array([-0.00914946, -0.00020783])
        """
        if isinstance(callbacks, t.Iterable):
            callbacks = _CallbackList(callbacks)
        else:
            callbacks = _CallbackList([callbacks])
        if cost_goal is not None:
            callbacks.append(_make_cost_goal_callback(cost_goal))
        generator = self.make_generator(x0, bounds=bounds, max_calls=max_calls)
        try:
            params = next(generator)
            while True:
                cost = func(params)
                if callbacks(self, params, cost):
                    break
                params = generator.send(cost)
        except StopIteration as exc:
            # This branch runs if `max_calls` is reached and the
            # generator is exhausted.
            params = exc.value
        return params


@t.overload
def optimize(
    func: t.Callable[[np.ndarray], float],
    x0: np.ndarray,  # pylint: disable=invalid-name
    *,
    max_calls: None = ...,
    cost_goal: t.Optional[float] = ...,
    callbacks: t.Union[Callback, t.Iterable[Callback]] = ...,
    bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = ...,
    gain: float = ...,
    oscillation_size: float = ...,
    oscillation_sampling: int = ...,
    decay_rate: float = ...,
) -> t.NoReturn:
    ...


@t.overload
def optimize(
    func: t.Callable[[np.ndarray], float],
    x0: np.ndarray,  # pylint: disable=invalid-name
    *,
    max_calls: int,
    cost_goal: t.Optional[float] = ...,
    callbacks: t.Union[Callback, t.Iterable[Callback]] = ...,
    bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = ...,
    gain: float = ...,
    oscillation_size: float = ...,
    oscillation_sampling: int = ...,
    decay_rate: float = ...,
) -> np.ndarray:
    ...


def optimize(
    func: t.Callable[[np.ndarray], float],
    x0: np.ndarray,  # pylint: disable=invalid-name
    *,
    max_calls: t.Optional[int] = None,
    cost_goal: t.Optional[float] = None,
    callbacks: t.Union[Callback, t.Iterable[Callback]] = (),
    bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
    gain: float = 0.2,
    oscillation_size: float = 0.1,
    oscillation_sampling: int = 10,
    decay_rate: float = 1.0,
) -> np.ndarray:
    """Run an optimization loop using ES.

    Args:
        x0: The initial set of parameters to suggest.
        max_calls: If passed, end the generator after this many
            steps.
        cost_goal: If passed, end optimization once this threshold is
            crossed. if :attr:`gain` is positive (the default), the cost
            must be less than *cost_goal* to end optimization. If
            :attr:`gain` is negative, the cost must be greater.
        callbacks: If passed, should be either a callback function or a
            list of them. They are called on each iteration with 3
            arguments: the :class:`ExtremumSeeker` instance, the current
            set of parameters, and the cost associated with these
            parameters. If any callback returns a truth-like value,
            optimization terminates.
        bounds: If passed, a tuple ``(lower, upper)`` of bounds
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
            :meth:`make_generator()` and :meth:`optimize()`.

    Returns:
        If *max_calls* has been supplied, this returns the final set
        of parameters. Otherwise, this method never returns.

    Example:

        >>> def cost_function(x):
        ...     return np.mean(x*x)
        >>> optimize(cost_function, x0=np.zeros(2), max_calls=10)
        array([-0.00914946, -0.00020783])
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


class _CallbackList(t.List[Callback]):
    """List of callbacks.

    Calling this list always calls all elements. It ends optimization if
    any of the elements does.

    Example:

        >>> def make_printer(i):
        ...     def printer(*args):
        ...         print(i)
        ...         return i == 2
        ...     return printer
        >>> cblist = _CallbackList(make_printer(i) for i in range(1, 4))
        >>> cblist(ExtremumSeeker(), np.zeros(2), 0.0)
        1
        2
        3
        True
    """

    def __call__(self, seeker: ExtremumSeeker, params: np.ndarray, cost: float) -> bool:
        # Collect bools into a list first to ensure that each callback
        # is called. `any()` stops on the first True it finds.
        # pylint: disable = use-a-generator
        return any([bool(cb(seeker, params, cost)) for cb in self])


def _make_cost_goal_callback(cost_goal: float) -> Callback:
    def _cost_goal(seeker: ExtremumSeeker, _: np.ndarray, cost: float) -> bool:
        return (cost < cost_goal) if seeker.gain > 0.0 else (cost > cost_goal)

    return _cost_goal


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
