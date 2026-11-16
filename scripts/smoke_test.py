"""Fast integrity check for the public reproducibility bundle.

This intentionally does not import GPU/model packages or rerun experiments.
It checks the files that a reviewer should be able to inspect immediately.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def json_files(path: Path):
    return sorted(path.rglob("*.json"))


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "manuscript" / "main.tex",
        ROOT / "manuscript" / "article.pdf",
        ROOT / "experiments" / "cem" / "probe_heldout.py",
        ROOT / "experiments" / "hicem" / "hicem_checks3.py",
        ROOT / "experiments" / "dreamsim" / "eval_b.py",
        ROOT / "experiments" / "goemotions" / "conditional_probe.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))

    result_root = ROOT / "results"
    result_count = sum(1 for p in result_root.rglob("*") if p.is_file())
    if result_count == 0:
        raise SystemExit("No bundled result files found")

    # Parse every compact JSON result. JSONL files are checked line by line.
    for path in json_files(result_root):
        with path.open(encoding="utf-8") as fh:
            json.load(fh)
    for path in sorted(result_root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"Invalid JSONL {path}:{line_no}: {exc}") from exc

    # The public tree must not accidentally contain training checkpoints.
    forbidden = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if forbidden:
        raise SystemExit("Checkpoint files must not be published: " + ", ".join(str(p) for p in forbidden))

    print(f"OK: {result_count} compact result files; all JSON/JSONL parsed successfully")


if __name__ == "__main__":
    main()
