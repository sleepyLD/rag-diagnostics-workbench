"""Remove absolute local paths from notebook outputs before committing.

Usage:
    python -m tools.clean_notebook_paths            # all notebooks/*.ipynb
    python -m tools.clean_notebook_paths --check    # only report, exit 1 if anything is found

Outputs such as "ROOT = D:\\...\\rag-workbench" or "已存到 D:\\...\\results\\..." reveal where the
project lives on your computer. This replaces the project root with <專案根目錄> and any other
user home folder with <HOME>. Only cell outputs are touched; code and markdown stay as they are.
Re-running a notebook brings the paths back, so run this again before each commit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    # ...\rag-workbench (Windows drive or POSIX), any separators, case-insensitive
    (re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:[^\\/\n\"']+[\\/])*?rag-workbench(?=[\\/\s\"'|]|$)", re.I), "<專案根目錄>"),
    (re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\n\"']+", re.I), "<HOME>"),
    (re.compile(r"/(?:home|Users)/[^/\n\"']+"), "<HOME>"),
]


def clean_text(s: str) -> tuple[str, int]:
    n = 0
    for pat, rep in PATTERNS:
        s, k = pat.subn(rep, s)
        n += k
    return s, n


def clean_value(v):
    """Recursively clean strings inside an output object. Returns (value, count)."""
    if isinstance(v, str):
        return clean_text(v)
    if isinstance(v, list):
        out, n = [], 0
        for x in v:
            x, k = clean_value(x)
            out.append(x)
            n += k
        return out, n
    if isinstance(v, dict):
        out, n = {}, 0
        for key, x in v.items():
            x, k = clean_value(x)
            out[key] = x
            n += k
        return out, n
    return v, 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="default: notebooks/*.ipynb")
    ap.add_argument("--check", action="store_true", help="report only, do not modify")
    args = ap.parse_args(argv)

    files = [Path(f) for f in args.files] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    total = 0
    for f in files:
        nb = json.loads(f.read_text(encoding="utf-8"))
        n = 0
        for cell in nb.get("cells", []):
            if "outputs" in cell:
                cell["outputs"], k = clean_value(cell["outputs"])
                n += k
        total += n
        print(f"{f.name}: {n} 處")
        if n and not args.check:
            f.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.check and total:
        print("仍有本機路徑，請執行：python -m tools.clean_notebook_paths")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
