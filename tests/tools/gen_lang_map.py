# tools/gen_lang_map.py
import json, sys
from pathlib import Path
fn = Path("./data/translation_data/combined_3pairs.json")
out = Path("./data/lang_pair_map.json")
data = json.loads(fn.read_text(encoding="utf-8"))
seen = {}
idx = 0
for item in data:
    lp = item.get("lang_pair", "unknown")
    if lp not in seen:
        seen[lp] = idx
        idx += 1
# ensure unknown present
if "unknown" not in seen:
    seen["unknown"] = idx
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", out, "with", len(seen), "entries")
