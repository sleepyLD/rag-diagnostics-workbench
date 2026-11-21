"""Download NeoQA from Hugging Face and save it as plain jsonl under data/raw/neoqa/ (git-ignored).

NeoQA (Glockner et al., ACL 2025 Findings) is built from *fictional* news timelines, so no model
can have seen the answers during pre-training. License CC BY-ND 4.0: the authors ask that the
decrypted data is not shared in plain text, so it must never be committed.

The Hugging Face repo uses a loading script, which `datasets` 4.x no longer supports. Use a
separate venv so the main environment is not touched:

    py -3.11 -m venv .venv-neoqa
    .\\.venv-neoqa\\Scripts\\python.exe -m pip install "datasets>=2.19,<4"
    .\\.venv-neoqa\\Scripts\\python.exe -m tools.datasets.download_neoqa --key 23456

The key (23456) is published in the NeoQA README. Default configs: `optimal-evidence` (each
question + the minimal sufficient articles, plus versions with articles removed) and `news`
(all articles, for building a retrieval collection). The full dataset is about 3.7 GB, mostly
`context-ablation`, which is not needed here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=int, required=True, help="decryption key from the NeoQA README")
    ap.add_argument("--configs", nargs="+", default=["optimal-evidence", "news"])
    ap.add_argument("--out", default="data/raw/neoqa")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        ds = load_dataset("mglockner/neoqa", name=name, decryption_key=args.key, trust_remote_code=True)
        for split, d in ds.items():
            p = out / f"{name}__{split}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for row in d:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(f"{name}/{split}: {len(d)} rows -> {p}")
            print("   fields:", list(d.features))
            for col in ("category", "question_type", "answerable", "answer_type"):
                if col in d.column_names:
                    print(f"   {col}:", dict(Counter(map(str, d[col])).most_common(10)))
    print("\n完成。這些檔案不要提交到 git（.gitignore 已排除 data/raw/neoqa/）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
