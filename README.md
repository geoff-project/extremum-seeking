<!--
SPDX-FileCopyrightText: 2020-2023 CERN
SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
SPDX-FileNotice: All rights not expressly granted are reserved.

SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+
-->

# Extremum-Seeking Optimization and Control

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

This is an implementation of the extremum-seeking control algorithm as
described by [Scheinker et al.][]. The core idea is to let the parameters
oscillate around a center point and have the phase advance of the oscillation
depend on the cost function. ES spends more time where the cost function is low
and less time where it is high. This causes a slow drift in the parameter space
towards global minima.

[Scheinker et al.]: https://doi.org/10.1002/acs.3097

This package provides both an interface for numeric optimization (locating an
optimum) and for adaptive control (tracking a drifting/noisy optimum). It also
provides a coroutine-based interface, `ExtremumSeeker.make_generator()`, to
leave the control loop in the caller's hand.

This repository can be found online on CERN's [Gitlab][].

[Gitlab]: https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking/

## Table of Contents

[[_TOC_]]

## Installation

To install this package from the [Acc-Py Repository][], simply run the
following line while on the CERN network:

[Acc-Py Repository]: https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py

```shell
pip install cernml-extremum-seeking
```

To use the source repository, you must first install it as well:

```shell
git clone https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking.git
cd ./cernml-extremum-seeking/
pip install .
```

## Examples

Defining a cost function and creating an :class:`ExtremumSeeker` object:

```python
>>> rng = np.random.default_rng(0)
>>> loc = np.zeros(2)
>>> def cost_function(params: np.ndarray) -> float:
...     drift = rng.normal(scale=1e-2, size=loc.shape)
...     noise = rng.normal(scale=1e-3, size=loc.shape)
...     loc[:] += drift
...     cost = np.linalg.norm(loc + noise - params)
...     return cost
>>> seeker = ExtremumSeeker(oscillation_size=0.1)
```

Executing a single control step:

```python
>>> x0 = rng.normal(0.1, size=loc.shape)
>>> seeker.calc_next_step(x0, cost=cost_function(x0), step=0)
array([0.26159863, 0.03066484])
```

Creating a generator that receives cost values and yields the next
parameter to evaluate:

```python
>>> gen = seeker.make_generator(x0)
>>> cost = None
>>> for i in range(10):
...     it = gen.send(cost)
...     cost = cost_function(it.params)
>>> it.params
array([ 0.16964995, -0.09272651])
```

Running an optimization loop:

```python
>>> res = seeker.optimize(cost_function, x0, max_calls=10)
>>> print(res)
params: array([ 0.16998328, -0.09349066])
  cost: 0.1895720815951993
   nit: 10
```


Running an optimization loop until the cost is sufficiently small:

```python
>>> res = seeker.optimize(cost_function, x0, cost_goal=0.01)
>>> cost_function(res.params)
0.01050409604837506
```

Passing a callback function to the optimization loop:

```python
>>> def printer(seeker: ExtremumSeeker, iteration: Iteration):
...     print("Cost:", iteration.cost)
>>> _ = seeker.optimize(cost_function, x0, max_calls=1, callbacks=printer)
Cost: 0.6215048967082203
```

Passing multiple callbacks, one of which ends the loop immediately by
returning :obj:`True`:

```python
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
```

The [Examples](/examples) directory contains more comprehensive example
programs.

Documentation
-------------

Inside the CERN network, you can read the package documentation on the [Acc-Py
documentation server][acc-py-docs]. The API is also documented via extensive
Python docstrings.

[acc-py-docs]: https://acc-py.web.cern.ch/gitlab/geoff/optimizers/cernml-extremum-seeking/

## Citation

To cite this package in a publication, you can use the following BibTeX
template:

```bibtex
@online{cernml-es,
    author={Nico Madysa and Verena Kain},
    title={CERNML Extremum Seeking},
    version={3.0.0},
    date={2023-06-12},
    organization={CERN},
    url={https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking/-/tags/v3.0.0},
    urldate={<whenever you've last verified your online sources>},
}
```

Changelog
---------

[See here](https://acc-py.web.cern.ch/gitlab/geoff/optimizers/cernml-extremum-seeking/docs/stable/changelog.html).

Stability
---------

This package uses [Semantic Versioning](https://semver.org/).

License
-------

Except as otherwise noted, this work is licensed under either of [GNU Public
License, Version 3.0 or later](LICENSES/GPL-3.0-or-later.txt), or [European
Union Public License, Version 1.2 or later](LICENSES/EUPL-1.2.txt), at your
option. See [COPYING](COPYING) for details.

Unless You explicitly state otherwise, any contribution intentionally submitted
by You for inclusion in this Work (the Covered Work) shall be dual-licensed as
above, without any additional terms or conditions.

For full authorship information, see the version control history.
