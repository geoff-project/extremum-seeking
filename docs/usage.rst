..
    SPDX-FileCopyrightText: 2020 - 2025 CERN
    SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Usage
=====

..
    >>> from numpy.typing import NDArray
    >>> from cernml.extremum_seeking import Iteration, Callback

First, define the *cost function* whose minimum you want to find and track:

    >>> import numpy as np
    >>> from cernml.extremum_seeking import ExtremumSeeker
    ...
    >>> rng = np.random.default_rng(0)
    >>> loc = np.zeros(2)
    >>> def cost_function(params: NDArray[np.floating]) -> float:
    ...     drift = rng.normal(scale=1e-2, size=loc.shape)
    ...     noise = rng.normal(scale=1e-3, size=loc.shape)
    ...     loc[:] += drift
    ...     cost = np.linalg.norm(loc + noise - params)
    ...     return cost

Then create an `.ExtremumSeeker` object:

    >>> seeker = ExtremumSeeker(oscillation_size=0.1)


Single-stepping the Algorithm
-----------------------------

Use `.calc_next_step()` to make a single control step:

    >>> x0 = rng.normal(0.1, size=loc.shape)
    >>> step = seeker.calc_next_step(x0, cost=cost_function(x0))
    >>> step
    Step(params=array([0.25875051, 0.01703471]), nit=1)

The `~.Step.params` attribute contains the next set of parameters suggested by
the algorithm. To continue, evaluate the cost function again and pass the
result back:

    >>> seeker.calc_next_step(step, cost=cost_function(step.params))
    Step(params=array([0.28462345, 0.03355959]), nit=2)

Using a Generator
-----------------

Alternatively, you can also call `.make_generator()` to create
a :term:`generator` that receives cost values and yields the next step:

    >>> gen = seeker.make_generator(x0)
    >>> step = next(gen)
    >>> for i in range(10):
    ...     cost = cost_function(step.params)
    ...     step = gen.send(cost)
    >>> step.params
    array([ 0.13778937, -0.0300518 ])

Running an Optimization Loop
----------------------------

You can use `~.ExtremumSeeker.optimize()` to run an optimization loop for
a fixed number of steps:

    >>> res = seeker.optimize(cost_function, x0, max_calls=10)
    >>> print(res)
    params: [ 0.17010622 -0.0937126 ]
      cost: 0.19667245
       nit: 10

For convenience, there's also a module-level function
`~cernml.extremum_seeking.optimize()` that allows you to skip creating an
`.ExtremumSeeker` in the first place.

Pass the *cost_goal* argument to run an optimization loop until the cost is
sufficiently small:

    >>> res = seeker.optimize(cost_function, x0, cost_goal=0.01)
    >>> round(cost_function(res.params), 6)
    np.float64(0.007859)

Using Callback Functions
------------------------

You can pass a callback function to the optimization loop:

    >>> def printer(seeker: ExtremumSeeker, iteration: Iteration):
    ...     print("Cost:", round(iteration.cost, 6))
    >>> _ = seeker.optimize(cost_function, x0, max_calls=1, callbacks=printer)
    Cost: 0.63442

You can also pass multiple callbacks. If any of them returns `True`, the
optimization loop ends. (However, for a given iteration, all callbacks are
called.)

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
