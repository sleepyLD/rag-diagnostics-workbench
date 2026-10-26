"""Check a Markdown document against its raw snapshot.

Usage:
    python -m tools.ingest.verify collections/real_docs/belkin_wch014_faq.md

Checks (exit code 1 if any FAIL):
  1. raw_sha256  - raw snapshot is unchanged since conversion
  2. numbers     - every number+unit in the raw article appears in the .md the
                   same number of times, and the .md has no numbers the source
                   lacks (human annotation blocks excluded)
  3. entries     - Q/A count in .md equals data rows in the source table
  4. sentences   - every source cell text appears in the .md (catches dropped text)
  5. images      - each listed image has a human annotation block

Human annotations go in the .md like this and are excluded from checks 2 and 4:

    <!-- manual:start image=sf335765-001_en_v2.png by=yourname date=2026-10-22 -->
    > [人工描述] ...
    <!-- manual:end -->
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from .common import (find_article, norm_text, number_tokens, region_text,
                     split_front_matter, strip_manual_blocks)


def parse_fm(fm: str) -> dict:
    meta, key = {}, None
    for line in fm.splitlines():
        if line.startswith("  - ") and key:
            meta.setdefault(key, []).append(line[4:].strip())
        elif ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            meta[key] = [] if val in ("", "[]") else val
    return meta


def run(md_path: Path) -> list[tuple[str, str, str]]:
    """Return list of (status, check, message)."""
    results = []
    fm_text, body = split_front_matter(md_path.read_text(encoding="utf-8"))
    fm = parse_fm(fm_text)
    raw_path = Path(fm["raw_snapshot"])
    if not raw_path.is_absolute() and not raw_path.exists():
        raw_path = md_path.parent / raw_path
    raw_bytes = raw_path.read_bytes()
    meta = json.loads(raw_path.with_suffix(".meta.json").read_text(encoding="utf-8"))

    # 1. snapshot integrity
    sha = hashlib.sha256(raw_bytes).hexdigest()
    if sha == fm.get("raw_sha256") == meta["sha256"]:
        results.append(("PASS", "raw_sha256", "raw snapshot unchanged"))
    else:
        results.append(("FAIL", "raw_sha256", "raw snapshot differs from the hash recorded at conversion"))

    region = find_article(BeautifulSoup(raw_bytes, "html.parser"), meta["profile"])
    src_text = region_text(region)
    body_auto, manual_blocks = strip_manual_blocks(body)

    # 2. numbers, both directions
    src_nums, md_nums = number_tokens(src_text), number_tokens(body_auto)
    missing = src_nums - md_nums
    extra = md_nums - src_nums
    if not missing and not extra:
        results.append(("PASS", "numbers", f"{sum(src_nums.values())} number tokens match"))
    else:
        if missing:
            results.append(("FAIL", "numbers", f"in source but missing/short in .md: {dict(missing)}"))
        if extra:
            results.append(("FAIL", "numbers", f"in .md but not in source: {dict(extra)}"))

    # 3 & 4. structure and sentence coverage (table-based profiles)
    md_norm = norm_text(body_auto)
    if region["table"] is not None:
        rows = []
        for tr in region["table"].find_all("tr"):
            cells = [norm_text(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
            non_empty = [c for c in cells if c]
            if not non_empty:
                continue
            if len(non_empty) == 1 and non_empty[0].isupper() and len(non_empty[0]) < 60:
                continue  # section header
            rows.append(non_empty)
        n_md = len(re.findall(r"^### ", body_auto, flags=re.M))
        if n_md == len(rows):
            results.append(("PASS", "entries", f"{n_md} Q/A entries = {len(rows)} source rows"))
        else:
            results.append(("FAIL", "entries", f".md has {n_md} Q/A entries, source has {len(rows)} rows"))

        md_flat = norm_text(md_norm.replace("\n", " "))
        dropped = []
        for cells in rows:
            for c in cells:
                # compare cell text with all whitespace removed, so line breaks don't matter
                if re.sub(r"\s", "", c) not in re.sub(r"\s", "", md_flat):
                    dropped.append(c[:80])
        if not dropped:
            results.append(("PASS", "sentences", "all source cell text present in .md"))
        else:
            results.append(("FAIL", "sentences", f"{len(dropped)} cell(s) not found verbatim: {dropped}"))

    # 5. images
    imgs = fm.get("images_needing_annotation") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    unannotated = []
    for entry in imgs:
        img_id = entry.split(" ", 1)[0]
        if not any(re.search(rf"image={re.escape(img_id)}(\s|-->)", b) for b in manual_blocks):
            unannotated.append(img_id)
    if not imgs:
        results.append(("PASS", "images", "no images in article"))
    elif not unannotated:
        results.append(("PASS", "images", f"{len(imgs)} image(s) annotated"))
    else:
        results.append(("FAIL", "images", f"{len(unannotated)} image(s) need a manual annotation block: {unannotated}"))

    placeholders = [b.split("\n", 1)[0] for b in manual_blocks if "TODO" in b]
    if placeholders:
        results.append(("FAIL", "manual", f"{len(placeholders)} annotation block(s) still contain TODO: {placeholders}"))

    # informational
    results.append(("INFO", "manual", f"{len(manual_blocks)} human annotation block(s) excluded from text checks"))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("md", nargs="+")
    args = ap.parse_args(argv)
    failed = False
    for p in args.md:
        print(f"== {p}")
        for status, check, msg in run(Path(p)):
            print(f"  [{status}] {check}: {msg}")
            failed |= status == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
