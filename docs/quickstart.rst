..
    SPDX-FileCopyrightText: 2020-2023 CERN
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Quickstart
==========

Installation
------------

To install this package from the `Acc-Py Repository`_, simply run the
following line while on the CERN network:

.. _Acc-Py Repository:
   https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py

.. code-block:: shell-session

    $ pip install cernml-extremum-seeking

To use the source repository, you must first install it as well:

.. code-block:: shell-session

    $ git clone https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking.git
    $ cd ./cernml-extremum-seeking/
    $ pip install .

Examples
--------

See the :doc:`/examples/index` page and the :doc:`/api`.

Citation
--------

To cite this package in a publication, you can use the following BibTeX
template:

.. code-block:: bibtex

    @online{cernml-es,
        author={Nico Madysa and Verena Kain},
        title={CERNML Extremum Seeking},
        version={3.0.0},
        date={2023-06-12},
        organization={CERN},
        url={https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking/-/tags/v3.0.0},
        urldate={<whenever you've last verified your online sources>},
    }

License
-------

Except as otherwise noted, this work is licensed under either of `GNU Public
License, Version 3.0 or later <GPL-3.0-or-later>`_, or `European
Union Public License, Version 1.2 or later <EUPL-1.2>`_, at your
option. See COPYING_ for details.

Unless You explicitly state otherwise, any contribution intentionally submitted
by You for inclusion in this Work (the Covered Work) shall be dual-licensed as
above, without any additional terms or conditions.

For full authorship information, see the version control history.

.. _GPL-3.0-or-later: https://www.gnu.org/licenses/gpl-3.0.txt
.. _EUPL-1.2: https://joinup.ec.europa.eu/page/eupl-text-11-12
.. _COPYING: https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking/-/blob/master/COPYING
