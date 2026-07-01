#!/usr/bin/env python3
"""Build english.tex from preamble + new V9 body."""

ENGLISH = r'G:\mimo_project\circuit_ocr\arxiv_template\english.tex'

with open(ENGLISH, 'r', encoding='utf-8') as f:
    content = f.read()

doc_start = content.find(r'\begin{document}')
preamble = content[:doc_start]

with open(r'G:\mimo_project\circuit_ocr\arxiv_template\english_body_v9.tex', 'r', encoding='utf-8') as f:
    new_body = f.read()

content = preamble + new_body

with open(ENGLISH, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
for n in ['2,555', '0.7760', '0.8257', 'V8-Fixed', 'E1--E6', '44.4']:
    print(f'{n}: {content.count(n)}')
print(f'Braces: {content.count("{")} vs {content.count("}")} -> {"OK" if content.count("{")==content.count("}") else "MISMATCH"}')
print(f'Total: {len(content)} chars')
print('english.tex built successfully!')
