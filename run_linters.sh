#!/usr/bin/env sh

# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# Run all linters used by this project. This script is used in CI.
# Its main use is that it runs _all_ linters, even if one of them errors.

reuse lint

exit_code=$?
for cmd in "mypy" "black --check" "isort --check" "pylint"; do
  echo "Running $cmd ..."
  python -m $cmd "$@"
  exit_code="$((exit_code | $?))"
done

exit "$exit_code"
