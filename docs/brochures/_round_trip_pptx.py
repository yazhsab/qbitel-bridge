"""
Round-trip a PPTX through python-pptx's own save path.

This forces python-pptx to regenerate every XML part and rebuild the ZIP from
scratch with its canonical output. Fixes a class of validation failures where
the source ZIP had non-standard timestamps, CRC quirks, or unusual file
ordering that strict importers (notably Google Drive / Slides) silently
reject.

Usage:
    python3 _round_trip_pptx.py INPUT.pptx OUTPUT.pptx
"""
from __future__ import annotations

import sys
from pptx import Presentation


def round_trip(src: str, dst: str) -> int:
    prs = Presentation(src)
    n = len(prs.slides)
    prs.save(dst)
    return n


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    n = round_trip(sys.argv[1], sys.argv[2])
    print(f'round-tripped: {sys.argv[1]} -> {sys.argv[2]}  ({n} slides)')
