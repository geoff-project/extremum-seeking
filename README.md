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

[Gitlab]: https://gitlab.cern.ch/be-op-ml-optimization/cernml-extremum-seeking/

## Installation

To install this package from the [Acc-Py Repository][], simply run the
following line while on the CERN network:

```shell
pip install cernml-extremum-seeking
```

To use the source repository, you must first install it as well:

```shell
git clone https://gitlab.cern.ch/be-op-ml-optimization/cernml-extremum-seeking.git
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
>>> x0 = rng.normal(0.1, size=loc.sahape)
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
