#!/usr/bin/env sh
# Run once after cloning: sh scripts/setup-git-filter.sh
# Requires: uv tool install nb-clean

REPO_ROOT="$(git rev-parse --show-toplevel)"

git config filter.nb-clean.clean "$REPO_ROOT/scripts/nb-clean.sh"
git config filter.nb-clean.smudge cat
git config filter.nb-clean.required false

echo "nb-clean filter configured."