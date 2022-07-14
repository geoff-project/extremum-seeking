#!/usr/bin/env sh

### Run all linters used by this project. This script is used in CI.
### Its main use is that it runs _all_ linters, even if one of them errors.

exit_code=0
for cmd in "mypy" "black --check" "isort --check" "pylint"; do
  echo "Running $cmd ..."
  python -m $cmd "$@"
  exit_code="$((exit_code + $?))"
done

exit "$exit_code"
