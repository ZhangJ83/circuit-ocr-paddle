"""Topology-aware evaluation for circuit OCR outputs.
Extracts component references and types from netlist text,
computes component F1 and type accuracy beyond pure text NED.
"""
import re, json, sys
from pathlib import Path

RE_COMPONENT = re.compile(r'\b([A-Z]+)(\d+)\b')
RE_NET = re.compile(r'\b([A-Z][A-Z0-9_]{1,10})\b')

def extract_components(text):
    refs = []
    seen = set()
    for m in RE_COMPONENT.finditer(text):
        prefix, num, full = m.group(1), m.group(2), m.group(0)
        if full not in seen:
            refs.append((prefix, num, full))
            seen.add(full)
    return refs

def topology_metrics(pred_text, ref_text):
    pred_comps = extract_components(pred_text)
    ref_comps = extract_components(ref_text)
    pred_refs = set(c[2] for c in pred_comps)
    ref_refs = set(c[2] for c in ref_comps)
    intersection = pred_refs & ref_refs
    comp_precision = len(intersection) / max(len(pred_refs), 1)
    comp_recall = len(intersection) / max(len(ref_refs), 1)
    comp_f1 = 2 * comp_precision * comp_recall / max(comp_precision + comp_recall, 1e-9)
    ref_prefix_map = {ref: prefix for prefix, _, ref in ref_comps}
    correct_type = sum(1 for prefix, _, ref in pred_comps if ref in ref_prefix_map and prefix == ref_prefix_map[ref])
    type_acc = correct_type / max(len(intersection), 1)
    return {'comp_f1': round(comp_f1,4), 'comp_precision': round(comp_precision,4),
            'comp_recall': round(comp_recall,4), 'type_accuracy': round(type_acc,4),
            'pred_count': len(pred_refs), 'ref_count': len(ref_refs), 'matched': len(intersection)}

def evaluate_file(jsonl_path, limit=None):
    with open(jsonl_path, encoding='utf-8') as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if limit: samples = samples[:limit]
    total = {'comp_f1':0,'comp_precision':0,'comp_recall':0,'type_accuracy':0,
             'pred_count':0,'ref_count':0,'matched':0}
    valid = 0
    for s in samples:
        pred = s.get('prediction','')
        ref = s.get('label','')
        if not pred:
            total['ref_count'] += len(set(c[2] for c in extract_components(ref)))
            continue
        m = topology_metrics(pred, ref)
        for k in total: total[k] += m[k]
        valid += 1
    n = max(valid,1)
    return {k: round(v/n,4) if k.startswith('avg_') or k in ('comp_f1','comp_precision','comp_recall','type_accuracy') else v
            for k,v in {
        'samples': len(samples), 'valid': valid,
        'avg_comp_f1': round(total['comp_f1']/n,4),
        'avg_comp_precision': round(total['comp_precision']/n,4),
        'avg_comp_recall': round(total['comp_recall']/n,4),
        'avg_type_accuracy': round(total['type_accuracy']/n,4),
        'total_pred': total['pred_count'], 'total_ref': total['ref_count'],
        'total_matched': total['matched']
    }.items()}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    if args.verbose:
        with open(args.results, encoding='utf-8') as f:
            samples = [json.loads(l) for l in f if l.strip()]
        if args.limit: samples = samples[:args.limit]
        for i,s in enumerate(samples):
            m = topology_metrics(s.get('prediction',''), s.get('label',''))
            print(f"[{i+1}/{len(samples)}] F1={m['comp_f1']:.3f} P={m['comp_precision']:.3f} R={m['comp_recall']:.3f} type_acc={m['type_accuracy']:.3f} pred={m['pred_count']} ref={m['ref_count']} matched={m['matched']}")
    r = evaluate_file(args.results, args.limit)
    print(f"\n===== Topology: {Path(args.results).name} =====")
    for k,v in r.items(): print(f"  {k}: {v}")
