"""Shared helpers for the web-page ingestion tools.

The key idea: the *same* function decides which part of a page is "the article",
so snapshot / html_to_md / verify always talk about the same region.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from bs4 import BeautifulSoup, Tag

TOOL_VERSION = "0.1.0"

# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

def norm_text(s: str) -> str:
    """Normalise unicode width/dashes/spaces so trivial formatting differences
    do not count as content differences."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace(" ", " ")
    s = re.sub(r"[*_`]+", "", s)          # markdown emphasis
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# number, optional decimal / thousands separator, optional unit right after it
_NUM_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:[.,]\d+)*)\s*"
    r"(°C|°F|mAh|Wh|Hz|kHz|mW|cm|mm|in|lb|kg|oz|g|V|A|W|%)?(?![A-Za-z0-9])",
)


def number_tokens(text: str) -> Counter:
    """Multiset of 'number+unit' tokens, e.g. '20V', '3.25A', '0°C', '2,500'."""
    text = norm_text(text)
    out: Counter = Counter()
    for m in _NUM_RE.finditer(text):
        num, unit = m.group(1), m.group(2) or ""
        out[f"{num}{unit}"] += 1
    return out


# --------------------------------------------------------------------------
# Locating the article region
# --------------------------------------------------------------------------

class RegionNotFound(Exception):
    pass


def image_entry(img: Tag) -> str:
    """'<id> <src>' where id is a filename-like alt text if present, else the URL's last
    path segment. Annotation blocks refer to images by this id."""
    src = img["src"]
    alt = (img.get("alt") or "").strip()
    if re.fullmatch(r"[\w.\-]+\.(png|jpe?g|gif|webp|svg)", alt, flags=re.I):
        img_id = alt
    else:
        img_id = src.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or src
    return f"{img_id} {src}"


def find_article(soup: BeautifulSoup, profile: str) -> dict:
    """Return {'title': str, 'table': Tag|None, 'container': Tag|None, 'images': [src,...]}.

    profile:
      'belkin-faq' : <h1> followed by a Q/A <table> (Belkin support-article pages)
      'generic'    : <main>/<article>, else <body> minus nav/header/footer
    """
    if profile == "belkin-faq":
        h1 = soup.find("h1")
        if h1 is None:
            raise RegionNotFound("no <h1> found - page may be rendered by JavaScript")
        table = h1.find_next("table")
        if table is None:
            raise RegionNotFound("no <table> after <h1> - layout changed or JS-rendered")
        # images between the title and the end of the table belong to the article
        imgs = []
        for el in h1.find_all_next(["img", "table"]):
            if el is table:
                break
            if el.name == "img" and el.get("src"):
                imgs.append(el)
        imgs += [img for img in table.find_all("img") if img.get("src")]
        images = [image_entry(i) for i in imgs]
        return {"title": norm_text(h1.get_text(" ")), "table": table,
                "container": None, "images": images}

    if profile == "generic":
        container = soup.find("main") or soup.find("article") or soup.body
        if container is None:
            raise RegionNotFound("no <main>, <article> or <body>")
        for junk in container.find_all(["nav", "header", "footer", "script",
                                        "style", "noscript", "form"]):
            junk.decompose()
        h1 = container.find("h1") or soup.find("title")
        return {"title": norm_text(h1.get_text(" ")) if h1 else "",
                "table": None, "container": container,
                "images": [image_entry(i) for i in container.find_all("img") if i.get("src")]}

    raise ValueError(f"unknown profile: {profile}")


def region_text(region: dict) -> str:
    """Plain text of the article region (used by verify as ground truth)."""
    parts = [region["title"]]
    node = region["table"] if region["table"] is not None else region["container"]
    parts.append(node.get_text(" "))
    return norm_text(" ".join(parts))


# --------------------------------------------------------------------------
# Markdown front matter (tiny, dependency-free)
# --------------------------------------------------------------------------

def dump_front_matter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            lines += [f"  - {item}" for item in v] if v else []
            if not v:
                lines[-1] = f"{k}: []"
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def split_front_matter(md: str) -> tuple[str, str]:
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        if end != -1:
            return md[4:end], md[end + 5:]
    return "", md


MANUAL_START = "<!-- manual:start"
MANUAL_END = "<!-- manual:end -->"


def strip_manual_blocks(md_body: str) -> tuple[str, list[str]]:
    """Remove human-written annotation blocks; return (body_without, blocks)."""
    blocks = []
    pattern = re.compile(re.escape(MANUAL_START) + r".*?" + re.escape(MANUAL_END), re.S)
    for m in pattern.finditer(md_body):
        blocks.append(m.group(0))
    return pattern.sub("", md_body), blocks
