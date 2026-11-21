"""Derive <dataset>_named from a Fin-RATE dataset: add corpus fields company / ticker / form / filed / section parsed
from the chunk title (TICKER_DATE_FORM_NN_..._section) plus a ticker -> company-name map, for contextual headers.
A name is used only if its first distinctive word occurs in that ticker's own filings (else the ticker alone).
    python -m tools.bench.add_finrate_meta --dataset finrate_xdoc"""
import argparse, hashlib, json, re, shutil
from collections import defaultdict
from tools.bench.common import ROOT

NAMES = {"AGYS": "Agilysys Inc", "ALP.PQ": "Alpine Summit Energy Partners Inc", "AMNB": "American National Bankshares Inc",
         "BSRR": "Sierra Bancorp", "BTM": "Bitcoin Depot Inc", "CDZI": "Cadiz Inc", "CGC": "Canopy Growth Corp",
         "DBO": "Invesco DB Oil Fund", "DERM": "Journey Medical Corp", "EACO": "EACO Corp", "EQH": "Equitable Holdings Inc",
         "EXP": "Eagle Materials Inc", "FHN": "First Horizon Corp", "FIGS": "FIGS Inc", "FIX": "Comfort Systems USA Inc",
         "FLUT": "Flutter Entertainment plc", "GEN": "Gen Digital Inc", "GILD": "Gilead Sciences Inc", "GS": "Goldman Sachs Group Inc",
         "GSFI": "Green Stream Holdings Inc", "GSK": "GSK plc", "GXXM": "GEX Management Inc", "HIGR": "Hi-Great Group Holding Co",
         "HSBC": "HSBC Holdings plc", "HUSA": "Houston American Energy Corp", "IVFH": "Innovative Food Holdings Inc",
         "KSEZ": "Kinetic Seas Inc", "LANC": "Lancaster Colony Corp", "LW": "Lamb Weston Holdings Inc", "MO": "Altria Group Inc",
         "NOVA": "Sunnova Energy International Inc", "NRT": "North European Oil Royalty Trust", "NWN": "Northwest Natural Gas Co",
         "NXPL": "NextPlat Corp", "ONC": "BeOne Medicines Ltd (BeiGene)", "PENN": "Penn National Gaming Inc", "PSX": "Phillips 66",
         "REEMF": "Rare Element Resources Ltd", "RL": "Ralph Lauren Corp", "UFI": "Unifi Inc", "VCYT": "Veracyte Inc",
         "VFC": "V.F. Corp", "VLO": "Valero Energy Corp"}
GENERIC = {"inc", "corp", "co", "ltd", "plc", "group", "holdings", "the", "company", "fund", "trust", "energy", "international"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fiscal-year", action="store_true", help="add fiscal_year (questions say 'in 2021' about the 10-K filed in 2022)")
    ap.add_argument("--suffix", default="_named")
    ap.add_argument("--quarter", action="store_true", help="10-Q: fiscal quarter guessed from the filing month (filed Apr-Jun -> Q1, Jul-Sep -> Q2, Oct-Dec -> Q3)")
    a = ap.parse_args()
    src, dst = ROOT / "bench" / "data" / a.dataset, ROOT / "bench" / "data" / f"{a.dataset}{a.suffix}"
    dst.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(l) for l in open(src / "corpus.jsonl", encoding="utf-8")]
    text_by_tk = defaultdict(list)
    for d in docs:
        text_by_tk[d["title"].split("_")[0]].append(d["text"][:3000].lower())
    verified = {}
    for tk, name in NAMES.items():
        key = next((w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in GENERIC and len(w) > 2), name.lower())
        verified[tk] = name if any(key in t for t in text_by_tk.get(tk, [])) else None
    with open(dst / "corpus.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            p = d["title"].split("_")
            tk, filed, form = p[0], (p[1] if len(p) > 1 else ""), (p[2] if len(p) > 2 else "")
            section = " ".join(p[4:]).replace("special item unknown", "").replace("item unknown", "").strip() if len(p) > 4 else ""
            name = verified.get(tk)
            qtr = ""
            if a.quarter and form == "10-Q" and re.match(r"\d{4}-\d{2}", filed):
                qtr = {4: "Q1", 5: "Q1", 6: "Q1", 7: "Q2", 8: "Q2", 9: "Q2", 10: "Q3", 11: "Q3", 12: "Q3"}.get(int(filed[5:7]), "")
            fy = ""
            if a.fiscal_year and re.match(r"\d{4}-\d{2}", filed):
                y, mth = int(filed[:4]), int(filed[5:7])
                # annual reports / proxies describe the fiscal year that ended before filing (10-K filed Mar 2022 -> FY2021)
                fy = str(y - 1) if form in ("10-K", "20-F", "DEF", "40-F") and mth <= 6 else str(y)
            f.write(json.dumps({**d, "company": f"{name} ({tk})" if name else tk, "form": form, "filed": filed, "section": section, **({"fiscal_year": fy} if a.fiscal_year else {}),
                               **({"quarter": qtr} if a.quarter else {})},
                               ensure_ascii=False) + "\n")
    for p in src.glob("queries_*.jsonl"):
        shutil.copyfile(p, dst / p.name)
    m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    m.update({"name": dst.name, "derived_from": a.dataset, "meta_fields": ["company", "form", "filed", "section"],
              "names_verified": {k: bool(v) for k, v in verified.items()},
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.glob("*.jsonl")}})
    (dst / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"dataset": dst.name, "n": len(docs), "unverified": sorted(k for k, v in verified.items() if not v)}))


if __name__ == "__main__":
    main()
