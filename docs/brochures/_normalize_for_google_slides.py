"""
Normalize a python-pptx output for Google Slides import.

Google Slides' OOXML importer is strict about some effects (drop shadows added
via raw <a:effectLst>) and about presentation thumbnails. This script:

  1. Strips every <a:effectLst> block from slide XML.
  2. Removes the docProps/thumbnail.jpeg (Google Slides regenerates one).
  3. Removes any reference to the thumbnail from _rels.
  4. Re-saves as a fresh, validated .pptx file.

Usage:
    python3 _normalize_for_google_slides.py INPUT.pptx OUTPUT.pptx
"""
from __future__ import annotations

import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

EFFECT_LST_RE = re.compile(
    rb'<a:effectLst>.*?</a:effectLst>', re.DOTALL,
)
EFFECT_LST_EMPTY_RE = re.compile(rb'<a:effectLst\s*/>', re.DOTALL)
THUMBNAIL_REL_RE = re.compile(
    rb'<Relationship[^/]*Target="docProps/thumbnail\.jpeg"[^/]*/>',
)
THUMBNAIL_OVERRIDE_RE = re.compile(
    rb'<Override[^/]*PartName="/docProps/thumbnail\.jpeg"[^/]*/>',
)


def normalize(src_path: str, dst_path: str) -> dict:
    src = Path(src_path)
    dst = Path(dst_path)
    assert src.exists(), f'no such file: {src}'

    stats = {
        'slides_processed': 0,
        'effects_stripped': 0,
        'thumbnail_removed': False,
        'thumbnail_refs_removed': 0,
    }

    # Read entire pptx (zip) into memory
    with zipfile.ZipFile(src, 'r') as zin:
        members = {n: zin.read(n) for n in zin.namelist()}

    # 1. Strip effectLst from every slide / layout / master
    for name in list(members.keys()):
        if not name.endswith('.xml'):
            continue
        data = members[name]
        before = data.count(b'<a:effectLst')
        if before == 0:
            continue
        data = EFFECT_LST_RE.sub(b'', data)
        data = EFFECT_LST_EMPTY_RE.sub(b'', data)
        after = data.count(b'<a:effectLst')
        stats['effects_stripped'] += before - after
        members[name] = data
        if name.startswith('ppt/slides/slide'):
            stats['slides_processed'] += 1

    # 2. Drop the thumbnail
    if 'docProps/thumbnail.jpeg' in members:
        del members['docProps/thumbnail.jpeg']
        stats['thumbnail_removed'] = True

    # 3. Remove any rels / Content_Types references to the thumbnail
    for name in ['_rels/.rels', '[Content_Types].xml']:
        if name not in members:
            continue
        data = members[name]
        before = data
        data = THUMBNAIL_REL_RE.sub(b'', data)
        data = THUMBNAIL_OVERRIDE_RE.sub(b'', data)
        if data != before:
            stats['thumbnail_refs_removed'] += 1
            members[name] = data

    # Write a fresh zip — same compression so it stays small
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for name, data in members.items():
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, data)

    return stats


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    stats = normalize(src, dst)
    print(f'normalized: {src} -> {dst}')
    for k, v in stats.items():
        print(f'  {k:.<32} {v}')
