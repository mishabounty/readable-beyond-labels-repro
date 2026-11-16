#!/bin/bash
# Ridge-проба по набору z из свипа, затем удаление этих z (место на диске); строки пробы остаются в
# results/probe_ridge_<набор>.jsonl. z удаляются, только если для каждого файла есть строка пробы.
PY=${PYTHON:-python}
cd "$(dirname "$0")"
for n in "$@"; do
  d=results/z/$n
  [ -d $d ] || continue
  $PY probe_ridge.py $n --no-white > /dev/null 2>&1
  $PY - "$n" <<'PYEOF'
import json, sys
from pathlib import Path
n = sys.argv[1]
d = Path("results/z") / n
f = Path(f"results/probe_ridge_{n}.jsonl")
done = {(r["method"], r["lambda"], r["seed"]) for r in map(json.loads, open(f))} if f.exists() else set()
files = sorted(d.glob("*_s*.npy"))
missing = []
for z in files:
    m, lam, s = z.stem.rsplit("_", 2)
    if (m, float(lam), int(s[1:])) not in done:
        missing.append(z.name)
if missing:
    print(n, "не удаляю, без пробы:", missing)
else:
    for z in files:
        z.unlink()
    print(n, "удалено z:", len(files))
PYEOF
done
