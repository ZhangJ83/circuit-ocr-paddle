"""Fix PaddleOCR-VL processor: change return_tensors='pt' to 'pd'."""
path = "/root/.cache/huggingface/modules/transformers_modules/PaddleOCR_hyphen_VL/processing_paddleocr_vl.py"
with open(path, 'r') as f:
    content = f.read()

count = content.count('return_tensors="pt"')
content = content.replace('return_tensors="pt"', 'return_tensors="pd"')

with open(path, 'w') as f:
    f.write(content)

print(f"Fixed {count} occurrences of return_tensors='pt' -> 'pd' in {path}")
