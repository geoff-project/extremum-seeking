.. SPDX-FileCopyrightText: 2020-2026 CERN
.. SPDX-FileCopyrightText: 2023-2026 GSI Helmholtzzentrum für Schwerionenforschung
.. SPDX-FileNotice: All rights not expressly granted are reserved.
..
.. SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

:tocdepth: 3

Changelog
=========

.. currentmodule:: cernml.extremum_seeking

Unreleased
----------

v4.2.2
^^^^^^

Other changes
~~~~~~~~~~~~~
- Harmonize docs theme of Geoff packages.


v4.x
----

v4.2.1
^^^^^^

Other changes
~~~~~~~~~~~~~
- Update project links to point at the new website https://geoff.docs.cern.ch/.
- The package is now released on PyPI.

v4.2.0
^^^^^^

Additions
~~~~~~~~~
- The ``oscillation_size`` argument of `ExtremumSeeker` and `optimize()` now accepts a per-dimension array of shape :samp:`({ndim},)` in addition to a scalar.
- `AdaptiveAmplitude`, a configurable schedule that makes `Step.amplitude` track the cost error :samp:`|cost - cost_target|` instead of the monotonic ``decay_rate`` decay. Pass an instance via the new ``adaptive_amplitude`` argument of `ExtremumSeeker` or `optimize()`.

v4.1.0
^^^^^^

Additions
~~~~~~~~~
- Compatibility with NumPy 2.0.

Bug fixes
~~~~~~~~~
- Missing argument in docs for `ExtremumSeeker.optimize()`.
- To avoid confusion, `.make_generator()` and `.calc_next_step()` now return `Step` objects *without* a *cost* attribute. Callbacks continue to receive the full `.Iteration` objects.
- The :term:`generator` returned by `.make_generator()` now returns a distinct `Step` object from each step.
- The *cost* attribute of `Iteration` objects yielded by `.make_generator()` is always `~numpy.nan`.

Other changes
~~~~~~~~~~~~~
- Update :doc:`COI example </examples/coi_example>` to :doc:`cernml-coi <coi:index>` 0.9.
- Move :doc:`/usage` into a dedicated section.

v4.0.0
^^^^^^

Breaking changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.7 and 3.8.
- Increase required NumPy version to 1.23.

Additions
~~~~~~~~~
- Where possible, functions now accept `typing.SupportsFloat` instead of `float`.

v3.x
----

v3.0.3
^^^^^^

Bug fixes
~~~~~~~~~
- Build both stable and development docs on Python 3.9.

v3.0.2
^^^^^^

Additions
~~~~~~~~~
- Publish this documentation.

v3.0.1
^^^^^^

Other changes
~~~~~~~~~~~~~
- Start using `pre-commit <https://pre-commit.com/>`_.
- Open-source the package by adding license information.
- Extend :file:`README.md`, fix broken link.
- Improve Mypy namespace package detection.
- Shorten :file:`.gitignore` file.

v3.0.0
^^^^^^

Breaking changes
~~~~~~~~~~~~~~~~
- When printing an `OptimizeResult`, the scalars ``cost`` and ``nit`` now also honor :external+np:func:`numpy.set_printoptions()`.

Additions
~~~~~~~~~
- Support for Python 3.9.

Other changes
~~~~~~~~~~~~~
- Change project URL.
- Replace :file:`setup.cfg` with :file:`pyproject.toml`

v2.x
----

v2.0.0
^^^^^^

Breaking changes
~~~~~~~~~~~~~~~~
- Change arguments passed to `Callback` from ``(ExtremumSeeker, ndarray, float)`` to ``(ExtremumSeeker, Iteration)``, where the latter contains the arguments that it replaces and some more information.
- Change signature of `~ExtremumSeeker.calc_next_step()` and `~ExtremumSeeker.make_generator()` to yield new type `Iteration` instead of bare arrays.
- Remove enum ``OptimizationStatus``
- Remove attributes/properties ``status``, ``success`` and ``message`` from `OptimizeResult`
- Rename `OptimizeResult` attribute ``x`` to ``params`` and ``fun`` to ``cost``.

Additions
~~~~~~~~~
- type alias `Bounds`
- dataclass `Iteration` to pass more information back and forth

Bug fixes
~~~~~~~~~
- Tweak the precise timing of callbacks and loop termination, ensure that ``params`` and ``cost`` are always in-sync.
- Remove `~typing.NoReturn` from `optimize()` signature since it ignores the impact of callbacks.

v1.x
----

v1.0.0
^^^^^^

- Initial release
