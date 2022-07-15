from __future__ import annotations

import typing as t

import numpy as np


class ExtremumSeeker:

    _W_MIN: float = 1.0
    _W_MAX: float = 1.75

    def __init__(
        self,
        *,
        decay_rate: float = 1.0,
        oscillation_size: float = 0.1,
        oscillation_sampling: int = 10,
        initial_amplitude: float = 1.0,
        gain: float = 0.2,
    ) -> None:
        if not gain > 0.0:
            raise ValueError(f"gain must be positive: {gain}")
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be between 0 and 1: {decay_rate}")
        self.initial_amplitude = initial_amplitude
        self.decay_rate = decay_rate
        self.oscillation_size = oscillation_size
        self.oscillation_sampling = oscillation_sampling
        self.gain = gain

    @property
    def time_step(self) -> float:
        return 2 * np.pi / (self.oscillation_sampling * self._W_MAX)

    # This function defines one step of the ES algorithm at iteration i
    def calc_next_step(
        self,
        params: np.ndarray,
        *,
        cost: float,
        time: float,
        amplitude: float,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        # ES step for each parameter
        params = np.asanyarray(params)
        [ndim] = params.shape
        w_es = np.linspace(self._W_MIN, self._W_MAX, ndim)
        phases = time * w_es + self.gain * cost
        a_es = w_es * self.oscillation_size**2
        next_params = params + amplitude * self.time_step * np.cos(phases) * np.sqrt(
            a_es * w_es
        )
        if bounds is not None:
            lower, upper = bounds
            _check_bounds(ndim, lower, upper)
            next_params = np.clip(next_params, lower, upper)
        return next_params

    def make_generator(
        self,
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
        max_calls: t.Optional[int] = None,
    ) -> t.Generator[np.ndarray, float, np.ndarray]:
        params = np.copy(x0)
        amplitude = self.initial_amplitude
        i = 0
        while max_calls is None or i < max_calls:
            cost = yield params
            if np.isnan(cost):
                raise ValueError(f"cost is NaN (not a number) after {i} ES step(s)")
            params = self.calc_next_step(
                params=params,
                cost=cost,
                time=i * self.time_step,
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
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
        max_calls: None = None,
    ) -> t.NoReturn:
        ...

    @t.overload
    def optimize(
        self,
        func: t.Callable[[np.ndarray], float],
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
        max_calls: int,
    ) -> np.ndarray:
        ...

    def optimize(
        self,
        func: t.Callable[[np.ndarray], float],
        x0: np.ndarray,  # pylint: disable=invalid-name
        *,
        bounds: t.Optional[t.Tuple[np.ndarray, np.ndarray]] = None,
        max_calls: t.Optional[int] = None,
    ) -> np.ndarray:
        generator = self.make_generator(x0, bounds=bounds, max_calls=max_calls)
        try:
            params = next(generator)
            while True:
                cost = func(params)
                params = generator.send(cost)
        except StopIteration as exc:
            params = exc.value
            return params


def optimize(
    func: t.Callable[[np.ndarray], float],
    x0: np.ndarray,  # pylint: disable=invalid-name
    max_calls: t.Optional[int] = None,
    *,
    decay_rate: float = 1.0,
    oscillation_size: float = 0.1,
    oscillation_sampling: int = 10,
    initial_amplitude: float = 1.0,
    gain: float = 0.2,
) -> np.ndarray:
    seeker = ExtremumSeeker(
        decay_rate=decay_rate,
        oscillation_size=oscillation_size,
        oscillation_sampling=oscillation_sampling,
        initial_amplitude=initial_amplitude,
        gain=gain,
    )
    return seeker.optimize(func=func, x0=x0, max_calls=max_calls)


def _check_bounds(ndim: int, lower: np.ndarray, upper: np.ndarray) -> None:
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
