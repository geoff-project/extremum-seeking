..
    SPDX-FileCopyrightText: 2020 - 2025 CERN
    SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Changelog
=========

.. currentmodule:: cernml.extremum_seeking

Unreleased
----------

- ADD: Compatibility with NumPy 2.0.
- FIX: Missing argument in docs for `.ExtremumSeeker.optimize()`.
- FIX: The :term:`generator` returned by `.ExtremumSeeker.make_generator()` now returns a new `.Iteration` object from each step.
- OTHER: Update :doc:`COI example </examples/coi_example>` to :doc:`cernml-coi <coi:index>` 0.9.
- OTHER: Move :doc:`/usage` into a dedicated section.

v4.0.0
------

- BREAKING: Drop support for Python 3.7 and 3.8.
- BREAKING: Increase required NumPy version to 1.23.
- ADD: Where possible, functions now accept `typing.SupportsFloat` instead of
  `float`.

v3.0.3
------

- FIX: Build both stable and development docs on Python 3.9.

v3.0.2
------

- ADD: Publish this documentation.

v3.0.1
------

- OTHER: Start using `pre-commit <https://pre-commit.com/>`_.
- OTHER: Open-source the package by adding license information.
- OTHER: Extend :file:`README.md`, fix broken link.
- OTHER: Improve Mypy namespace package detection.
- OTHER: Shorten :file:`.gitignore` file.

v3.0.0
------

- BREAKING: When printing an `OptimizeResult`, the scalars ``cost`` and ``nit`` now also honor :external+np:func:`numpy.set_printoptions()`.
- ADD: Support for Python 3.9.
- OTHER: Change project URL.
- OTHER: Replace :file:`setup.cfg` with :file:`pyproject.toml`

v2.0.0
------

- BREAKING: Change arguments passed to `Callback` from ``(ExtremumSeeker, ndarray, float)`` to ``(ExtremumSeeker, Iteration)``, where the latter contains the arguments that it replaces and some more information.
- BREAKING: Change signature of `~ExtremumSeeker.calc_next_step()` and `~ExtremumSeeker.make_generator()` to yield new type `Iteration` instead of bare arrays.
- BREAKING: Remove enum ``OptimizationStatus``
- BREAKING: Remove attributes/properties ``status``, ``success`` and ``message`` from `OptimizeResult`
- BREAKING: Rename `OptimizeResult` attribute ``x`` to ``params`` and ``fun`` to ``cost``.
- ADD: type alias `Bounds`
- ADD: dataclass `Iteration` to pass more information back and forth
- FIX: Tweak the precise timing of callbacks and loop termination, ensure that ``params`` and ``cost`` are always in-sync.
- FIX: Remove `~typing.NoReturn` from `optimize()` signature since it ignores the impact of callbacks.

v1.0.0
------

- Initial release
