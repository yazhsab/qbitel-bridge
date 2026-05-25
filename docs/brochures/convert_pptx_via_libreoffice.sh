#!/usr/bin/env bash
# Convert the original PPTX through LibreOffice's engine so Google Slides
# accepts it. LibreOffice rewrites the OOXML using its own (Google-friendly)
# serializer.
#
# Usage:
#   bash docs/brochures/convert_pptx_via_libreoffice.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${DIR}/QBITEL_TCN_Partnership_Brief.pptx"
DST_DIR="${DIR}"
LO=/Applications/LibreOffice.app/Contents/MacOS/soffice

if [ ! -x "$LO" ]; then
    echo "LibreOffice not found at $LO" >&2
    exit 2
fi

# Convert into a tmp name so we don't clobber the original
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

"$LO" --headless --convert-to pptx --outdir "$TMP_DIR" "$SRC" >/dev/null
RESULT="$TMP_DIR/$(basename "${SRC%.pptx}.pptx")"

if [ ! -f "$RESULT" ]; then
    echo "Conversion failed — no output file in $TMP_DIR" >&2
    ls -la "$TMP_DIR" >&2
    exit 3
fi

OUT="${DST_DIR}/QBITEL_TCN_Partnership_Brief_LO.pptx"
cp "$RESULT" "$OUT"
echo "Converted: $OUT"
ls -la "$OUT"
