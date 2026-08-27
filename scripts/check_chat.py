import json
with open("/root/models/official_models/PaddleOCR-VL/tokenizer_config.json") as f:
    d = json.load(f)
ct = d.get("chat_template", "NOT_FOUND")
print(repr(ct)[:600])
