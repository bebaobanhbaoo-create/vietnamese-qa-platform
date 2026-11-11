"""Build a CSV for checking answer faithfulness by hand.

Citation checks only prove that an answer points at real, relevant sources; they cannot see an
answer that misreads its source (e.g. "tối thiểu 14 tín chỉ" reported as "tối đa 14 tín chỉ").
This sheet puts each answer next to the text it cites so a reviewer can mark it.

    python scripts/faithfulness_sheet.py eval/results/answers_dev_baseline.answers.jsonl --n 20

Fill the `faithful` column with 1 (every claim is backed by the cited text), 0 (some claim is
wrong or unsupported) or leave it empty for refusals, then run with --score to summarise.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from huflit_qa.config import get_settings
from huflit_qa.indexing.manifest import load_manifest

FIELDS = ["id", "query", "answer", "cited_sources", "faithful", "notes"]


def load_chunks(index_dir: Path) -> dict[str, dict]:
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return {c["chunk_id"]: c for c in chunks}


def build(answers_path: Path, index_dir: Path, n: int, seed: int) -> Path:
    load_manifest(index_dir / "manifest.json")  # fail early on a wrong index directory
    chunks = load_chunks(index_dir)
    records = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line]
    answered = [r for r in records if not r["refused"]]
    sample = random.Random(seed).sample(answered, min(n, len(answered)))

    out = answers_path.with_name(answers_path.name.replace(".answers.jsonl", ".faithfulness.csv"))
    # utf-8-sig so Excel shows Vietnamese correctly.
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in sample:
            cited = []
            for n_, chunk_id in enumerate(r["shown"], 1):
                if chunk_id in r["cited"]:
                    c = chunks[chunk_id]
                    cited.append(f"[{n_}] {c.get('chunk_title') or c['doc_title']}\n{c['text'][:1200]}")
            writer.writerow(
                {
                    "id": r["id"],
                    "query": r["query"],
                    "answer": r["answer"],
                    "cited_sources": "\n\n".join(cited),
                }
            )
    return out


def score(sheet: Path) -> None:
    with open(sheet, encoding="utf-8-sig") as f:
        marks = [row["faithful"].strip() for row in csv.DictReader(f)]
    judged = [m for m in marks if m in ("0", "1")]
    if not judged:
        print("No rows marked yet.")
        return
    print(f"faithful: {judged.count('1')}/{len(judged)} = {judged.count('1') / len(judged):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=Path, help="*.answers.jsonl to sample from, or a filled *.csv with --score"
    )
    parser.add_argument(
        "--index", type=Path, help="Index the answers were generated with (default: settings)"
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.score:
        score(args.path)
        return
    print(build(args.path, args.index or get_settings().index_dir, args.n, args.seed))


if __name__ == "__main__":
    main()
