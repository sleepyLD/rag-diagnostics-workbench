"""Save a raw, untouched snapshot of a web page plus provenance metadata.

Usage:
    python -m tools.ingest.snapshot URL --doc-id belkin_wch014_faq --profile belkin-faq

Writes:
    raw/<doc-id>.html        exact bytes the server returned
    raw/<doc-id>.meta.json   url, final url, time, status, sha256, size
Never edit files in raw/. Everything downstream is regenerated from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .common import TOOL_VERSION, RegionNotFound, find_article

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--profile", default="generic", choices=["belkin-faq", "generic"])
    ap.add_argument("--out-dir", default="raw")
    ap.add_argument("--from-file", help="use a page you saved manually in a browser instead of downloading")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{args.doc_id}.html"
    meta_path = out / f"{args.doc_id}.meta.json"

    if html_path.exists():
        print(f"[stop] {html_path} already exists. Raw snapshots are never overwritten; "
              f"use a new --doc-id (e.g. add a date) if the page changed.", file=sys.stderr)
        return 2

    if args.from_file:
        data = Path(args.from_file).read_bytes()
        status, final_url, method = None, None, "manual-browser-save"
    else:
        r = requests.get(args.url, headers={"User-Agent": UA}, timeout=30)
        data, status, final_url, method = r.content, r.status_code, r.url, "http-get"
        if status != 200:
            print(f"[fail] HTTP {status} for {args.url}", file=sys.stderr)
            return 1

    # Check the article is really in the HTML (catches JavaScript-rendered pages)
    try:
        region = find_article(BeautifulSoup(data, "html.parser"), args.profile)
        print(f"[ok] article found: {region['title']!r}")
    except RegionNotFound as e:
        print(f"[fail] {e}\n       Open the page in a browser, Save As 'Web page, HTML only', "
              f"then rerun with --from-file <saved.html>.", file=sys.stderr)
        return 1

    html_path.write_bytes(data)
    meta = {
        "doc_id": args.doc_id,
        "source_url": args.url,
        "final_url": final_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "http_status": status,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "profile": args.profile,
        "tool_version": TOOL_VERSION,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] saved {html_path} ({len(data):,} bytes) and {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
