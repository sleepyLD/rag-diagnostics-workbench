"""verify.py must catch the kinds of mistakes a human or a converter makes."""
from pathlib import Path

import pytest

from tools.ingest import html_to_md, snapshot, verify

FIXTURE = Path(__file__).parent / "fixtures" / "belkin_faq_sample.html"
URL = "https://www.belkin.com/support-article/?articleNum=335765"
ANNOTATION = (
    "\n<!-- manual:start image=sf335765-001_en_v2.png by=tester date=2026-10-22 -->\n"
    "> [人工描述] 圖中由上而下為 USB-C 頂部、中間、底部與 USB-A，共 4 個埠。\n"
    "<!-- manual:end -->\n"
)


@pytest.fixture
def doc(tmp_path, monkeypatch):
    """Snapshot + convert + annotate the image: a clean, passing document."""
    monkeypatch.chdir(tmp_path)
    assert snapshot.main([URL, "--doc-id", "belkin", "--profile", "belkin-faq",
                          "--from-file", str(FIXTURE)]) == 0
    md = tmp_path / "collections" / "real_docs" / "belkin.md"
    assert html_to_md.main(["raw/belkin.html", "--out", str(md)]) == 0
    text = md.read_text(encoding="utf-8")
    md.write_text(text.replace("\n## PRODUCT INFORMATION", ANNOTATION + "\n## PRODUCT INFORMATION", 1),
                  encoding="utf-8")
    return md


def statuses(md):
    return {check: status for status, check, _ in verify.run(md) if status != "INFO"}


def edit(md, old, new):
    text = md.read_text(encoding="utf-8")
    assert old in text, f"test setup: {old!r} not in document"
    md.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_clean_document_passes(doc):
    assert set(statuses(doc).values()) == {"PASS"}


def test_navigation_and_footer_are_excluded(doc):
    body = doc.read_text(encoding="utf-8")
    assert "Retractable GaN Wall Charger 67W" not in body   # nav
    assert "save 15%" not in body                          # footer


def test_section_headers_and_line_breaks(doc):
    body = doc.read_text(encoding="utf-8")
    assert "## FUNCTIONALITY" in body
    assert "28V / 5A / 140W\nUSB-C PPS:" in body


def test_altered_number_is_caught(doc):
    edit(doc, "28V / 5A / 140W", "24V / 5A / 140W")
    assert statuses(doc)["numbers"] == "FAIL"


def test_deleted_spec_line_is_caught(doc):
    edit(doc, "20V / 5A / 100W\n", "")
    s = statuses(doc)
    assert s["numbers"] == "FAIL" and s["sentences"] == "FAIL"


def test_dropped_entry_is_caught(doc):
    edit(doc, "### What is the warranty period of the Belkin WCH014?\n\n"
              "The Belkin WCH014 has a 2-year limited warranty.\n", "")
    assert statuses(doc)["entries"] == "FAIL"


def test_invented_number_is_caught(doc):
    edit(doc, "Weight: 0.76 lb", "Weight: 0.76 lb (345 g)")
    assert statuses(doc)["numbers"] == "FAIL"


def test_silently_fixing_source_typo_is_caught(doc):
    # The source says 0-20 °C; 'correcting' it to 40 °C must not pass unnoticed.
    edit(doc, "Between 0°C to 20°C", "Between 0°C to 40°C")
    assert statuses(doc)["numbers"] == "FAIL"


def test_numbers_inside_manual_block_are_allowed(doc):
    edit(doc, "共 4 個埠。", "共 4 個埠，頂部埠標示 140W。")
    assert statuses(doc)["numbers"] == "PASS"


def test_missing_image_annotation_is_caught(doc):
    edit(doc, "image=sf335765-001_en_v2.png", "image=wrong.png")
    assert statuses(doc)["images"] == "FAIL"


def test_placeholder_annotation_is_caught(doc):
    edit(doc, "> [人工描述] 圖中", "> [人工描述] TODO 圖中")
    assert statuses(doc)["manual"] == "FAIL"


def test_tampered_raw_snapshot_is_caught(doc):
    raw = doc.parents[2] / "raw" / "belkin.html"
    raw.write_bytes(raw.read_bytes() + b"<!-- edited -->")
    assert statuses(doc)["raw_sha256"] == "FAIL"


def test_snapshot_refuses_to_overwrite(doc, tmp_path):
    assert snapshot.main([URL, "--doc-id", "belkin", "--profile", "belkin-faq",
                          "--from-file", str(FIXTURE)]) == 2


def test_js_rendered_page_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shell = tmp_path / "shell.html"
    shell.write_text("<html><body><div id='app'></div><script>load()</script></body></html>")
    assert snapshot.main([URL, "--doc-id", "x", "--profile", "belkin-faq",
                          "--from-file", str(shell)]) == 1
    assert not (tmp_path / "raw" / "x.html").exists()
