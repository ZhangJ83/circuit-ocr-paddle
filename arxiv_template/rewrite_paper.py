#!/usr/bin/env python3
"""Rewrite template.tex completely with latest V8-Fixed progress."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TEMPLATE = r'G:\mimo_project\circuit_ocr\arxiv_template\template.tex'

with open(TEMPLATE, 'r', encoding='utf-8') as f:
    content = f.read()

# Keep preamble (lines up to \begin{document})
doc_start = content.find(r'\begin{document}')
preamble = content[:doc_start]

# Read new body from separate file
with open(r'G:\mimo_project\circuit_ocr\arxiv_template\template_body_v9.tex', 'r', encoding='utf-8') as f:
    new_body = f.read()

content = preamble + new_body

with open(TEMPLATE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'template.tex rewritten! Total: {len(content)} chars')
