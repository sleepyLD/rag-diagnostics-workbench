"""Convert a raw snapshot into a clean Markdown document for the RAG collection.

Usage:
    python -m tools.ingest.html_to_md raw/belkin_wch014_faq.html \
        --out collections/real_docs/belkin_wch014_faq.md \
        --manufacturer Belkin --model WCH014

Only the article region is converted (navigation, menus and footer are dropped).
Images cannot be converted to text; they are listed in the front matter under
`images_needing_annotation`, and verify.py fails until each one has a
human-written annotation block.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from .common import (TOOL_VERSION, dump_front_matter, find_article, norm_text)


def cell_text(td: Tag) -> str:
    """Cell text with <br> turned into line breaks, list items kept on separate lines."""
    for br in td.find_all("br"):
        br.replace_with(NavigableString("\n"))
    lines = [norm_text(line) for line in td.get_text("").split("\n")]
    lines = [l for l in lines if l]
    return "\n".join(lines)


def faq_table_to_md(table: Tag) -> str:
    out = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        texts = [cell_text(c) for c in cells]
        non_empty = [t for t in texts if t]
        if not non_empty:
            continue
        # section header row: one non-empty cell, short, upper-case (e.g. "FUNCTIONALITY")
        if len(non_empty) == 1 and non_empty[0].isupper() and len(non_empty[0]) < 60:
            out.append(f"\n## {non_empty[0]}\n")
            continue
        q, a = texts[0], "\n".join(texts[1:]).strip()
        out.append(f"### {q}\n\n{a}\n")
    return "\n".join(out).strip() + "\n"


def generic_to_md(container: Tag) -> str:
    out = []
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.find_parent("table") is not None and el.name != "table":
            continue  # handled with its table
        if el.name in ("h1", "h2", "h3", "h4"):
            level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[el.name]
            out.append(f"\n{level} {norm_text(el.get_text(' '))}\n")
        elif el.name == "li":
            out.append(f"- {norm_text(el.get_text(' '))}")
        elif el.name == "p":
            t = norm_text(el.get_text(" "))
            if t:
                out.append(t + "\n")
        elif el.name == "table":
            for tr in el.find_all("tr"):
                cells = [norm_text(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
                out.append("| " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out).strip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_html")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manufacturer", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--version", default="", help="product revision, e.g. 2024 for EP-T4511")
    args = ap.parse_args(argv)

    raw = Path(args.raw_html)
    meta_file = raw.with_suffix(".meta.json")
    if not meta_file.exists():
        print(f"[fail] {meta_file} missing - create snapshots with snapshot.py", file=sys.stderr)
        return 1
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    soup = BeautifulSoup(raw.read_bytes(), "html.parser")
    region = find_article(soup, meta["profile"])

    if region["table"] is not None:
        body = faq_table_to_md(region["table"])
    else:
        body = generic_to_md(region["container"])

    fm = dump_front_matter({
        "doc_id": meta["doc_id"],
        "title": region["title"],
        "manufacturer": args.manufacturer,
        "model": args.model,
        "version": args.version,
        "source_url": meta["source_url"],
        "retrieved_at": meta["retrieved_at"],
        "raw_snapshot": raw.as_posix(),
        "raw_sha256": meta["sha256"],
        "converted_by": f"html_to_md {TOOL_VERSION} (profile={meta['profile']})",
        "images_needing_annotation": region["images"],
        "known_source_issues": [],
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fm + f"\n# {region['title']}\n\n" + body, encoding="utf-8")
    n_q = body.count("\n### ") + body.startswith("### ")
    print(f"[ok] wrote {out}: {n_q} Q/A entries, {len(region['images'])} image(s) to annotate")
    for entry in region["images"]:
        print(f"     image: {entry}")
    if region["images"]:
        print("     -> open each image, then add to the .md:\n"
              "        <!-- manual:start image=<id> by=<name> date=<YYYY-MM-DD> -->\n"
              "        > [人工描述] ...\n"
              "        <!-- manual:end -->")
    return 0


if __name__ == "__main__":
    sys.exit(main())
