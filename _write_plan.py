
import sys
sys.stdout.reconfigure(encoding='utf-8')

lines = []
lines.append('')
lines.append('---')
lines.append('')
lines.append('## Phase 2: Data Diversity + Regularization + Topology Eval')
lines.append('')

with open(r'G:\mimo_project\circuit_ocr\PHASE_PLAN.md', 'a', encoding='utf-8') as f:
    for line in lines:
        f.write(line + chr(10))
print('appended header')
