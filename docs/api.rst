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

    Lower and upper bounds for the search space.

    Bounds are specified as a tuple :samp:`({lower}, {upper})` with the same
    shape as the parameters.

.. type:: Callback
    :canonical: Callable[[ExtremumSeeker, Iteration], Optional[bool]]

    Signature of callbacks for `optimize`.

    Each callback is called at the end of each iteration with 2 arguments:
    the `ExtremumSeeker` instance and the current `Iteration`.

    If any callback returns any truth-like value, optimization terminates.
    If the return value is false-like (including `None`), optimization
    continues. This is so that a callback without return value never
    terminates the optimization.

.. autoclass:: OptimizeResult()

.. autoclass:: Iteration()
