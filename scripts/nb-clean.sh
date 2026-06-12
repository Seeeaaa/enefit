#!/usr/bin/env sh
set +e
if command -v nb-clean >/dev/null 2>&1; then
    nb-clean clean \
      --remove-empty-cells \
      --preserve-cell-outputs \
      --remove-all-notebook-metadata
else
    cat  # pass-through if nb-clean is not installed
fi
exit 0