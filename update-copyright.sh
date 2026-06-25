#!/bin/sh

# SPDX-FileCopyrightText: 2024-2026 CERN
# SPDX-FileCopyrightText: 2024-2026 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# Use https://github.com/fsfe/reuse-tool to add the current year to all
# copyright headers.

reuse annotate \
    --merge-copyrights \
    --fallback-dot-license \
    --license="GPL-3.0-or-later OR EUPL-1.2+" \
    --copyright="CERN" \
    --copyright="GSI Helmholtzzentrum für Schwerionenforschung" \
    --year="$(date +%Y)" \
    --template=geoff \
    --recursive \
    .
