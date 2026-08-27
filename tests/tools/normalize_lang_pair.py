# # tools/normalize_lang_pair.py
# import json
# from pathlib import Path
# p = Path("./data/translation_data/combined_3pairs.json")
# data = json.loads(p.read_text(encoding="utf-8"))
# def normalize(s):
#     if s is None:
#         return "unknown"
#     s = s.strip()
#     # 将 en_zh -> en->zh，同时保持 en->zh 不变
#     s = s.replace("_", "->").replace("-", "->")
#     return s

# for item in data:
#     item["lang_pair"] = normalize(item.get("lang_pair", None))
# p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
# print("normalized", len(data))
