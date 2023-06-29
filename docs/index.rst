..
    SPDX-FileCopyrightText: 2020-2023 CERN
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Extremum-Seeking Optimization and Control
=========================================

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

This is an implementation of the extremum-seeking control algorithm as
described by `Scheinker et al.`_. The core idea is to let the parameters
oscillate around a center point and have the phase advance of the oscillation
depend on the cost function. ES spends more time where the cost function is low
and less time where it is high. This causes a slow drift in the parameter space
towards global minima.

.. _Scheinker et al.: https://doi.org/10.1002/acs.3097

This package provides both an interface for numeric optimization (locating an
optimum) and for adaptive control (tracking a drifting/noisy optimum). It also
provides a coroutine-based interface,
`~cernml.extremum_seeking.ExtremumSeeker.make_generator()`, to leave the
control loop in the caller's hand.

This repository can be found online on CERN's Gitlab_.

.. _Gitlab: https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking/


.. toctree::
   :maxdepth: 2

   quickstart
   examples/index
   api
   changelog
