..
    SPDX-FileCopyrightText: 2020 - 2025 CERN
    SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Usage
=====

Defining a cost function and creating an `.ExtremumSeeker` object:

..
    >>> from numpy.typing import NDArray
    >>> from cernml.extremum_seeking import Iteration, Callback

.. code-block:: python

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
    >>> seeker = ExtremumSeeker(oscillation_size=0.1)

Executing a single control step:

    >>> x0 = rng.normal(0.1, size=loc.shape)
    >>> seeker.calc_next_step(x0, cost=cost_function(x0))
    Step(params=array([0.25875051, 0.01703471]), nit=1)

Creating a generator that receives cost values and yields the next
parameter to evaluate:

    >>> gen = seeker.make_generator(x0)
    >>> cost = None
    >>> for i in range(10):
    ...     it = gen.send(cost)
    ...     cost = cost_function(it.params)
    >>> it.params
    array([ 0.16964995, -0.09272651])

Running an optimization loop for a fixed number of steps:

    >>> res = seeker.optimize(cost_function, x0, max_calls=10)
    >>> print(res)
    params: [ 0.16998328 -0.09349066]
      cost: 0.18957208
       nit: 10


Running an optimization loop until the cost is sufficiently small:

    >>> res = seeker.optimize(cost_function, x0, cost_goal=0.01)
    >>> round(cost_function(res.params), 6)
    np.float64(0.010504)

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
