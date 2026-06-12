#!/usr/bin/env sh
set +e

# Buffer stdin — can only be read once
TMPFILE=$(mktemp)
cat > "$TMPFILE"

if command -v nb-clean >/dev/null 2>&1; then
    # If nb-clean fails at runtime — fall back to cat
    nb-clean clean \
      --remove-empty-cells \
      --preserve-cell-outputs \
      --remove-all-notebook-metadata < "$TMPFILE" || cat "$TMPFILE"
else
    cat "$TMPFILE"
fi

rm -f "$TMPFILE"
exit 0