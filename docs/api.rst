..
    SPDX-FileCopyrightText: 2020 - 2025 CERN
    SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

:tocdepth: 3

API Reference
=============

.. seealso::

    :doc:`/usage`
        Small example snippets that show-case how to use this package.
    :doc:`/examples/index`
        More comprehensive examples that cover entire programs.

.. automodule:: cernml.extremum_seeking
    :no-members:

Functional API
--------------

.. autofunction:: optimize

Class-based API
---------------

.. autoclass:: ExtremumSeeker

Helper Types
------------

.. type:: Bounds
    :canonical: tuple[NDArray[np.floating], NDArray[np.floating]]

    Type alias that describes the search space.

    Bounds are specified as a tuple :samp:`({lower}, {upper})` of arrays with
    the same shape as the parameters. This alias is used by `optimize()` and
    `ExtremumSeeker.make_generator()`.

.. type:: Callback
    :canonical: Callable[[ExtremumSeeker, Iteration], Optional[bool]]

    Type alias that describes the signature of optimization callbacks.

    Each callback is called *at the end* of each iteration with 2 arguments:
    the `ExtremumSeeker` instance and the current `Iteration` object.

    If *any* callback returns a :ref:`true-like <std:truth>` value,
    optimization terminates. If the return value is false-like (including
    `None`), optimization continues. This way, callbacks without a return
    statement continue the optimization by default.

    This alias is used by `optimize()` and `ExtremumSeeker.optimize()`.

    .. note::
        The `Iteration` object passed to callbacks is mutable. If one callback
        modifies it, all subsequent callbacks *in the same iteration* can
        observe this modification.

        However, each iteration is run with a distinct `Iteration` object. So
        modifications will not transfer to subsequent iterations.

.. autoclass:: OptimizeResult()

.. autoclass:: Step()

.. autoclass:: Iteration()
