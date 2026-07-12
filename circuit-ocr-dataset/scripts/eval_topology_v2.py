"""Topology-aware evaluation v2 — richer text-level metrics from flat netlist text.

The Phase-1 metric `component_f1` only checks whether a reference designator
(R1, C2, U3...) appears in both prediction and GT. A model that outputs "R1"
when the GT is "R1\\n100k" — but predicts the value as "10k" — still gets full
component_f1 credit. That overstates success.

This script parses the flat "refdes\\nvalue\\nrefdes\\nvalue ..." netlist text
into (refdes, value) pairs and computes three honest metrics:

  - comp_f1     : reference-designator set F1 (sanity-check vs Phase-1 number)
  - value_acc   : among refdes matched between pred and GT, fraction whose
                  value is also correct (case-insensitive, whitespace-ignored)
  - joint_f1    : (refdes, value) PAIR set F1 — a component counts only if both
                  its ID and value are right. joint_f1 <= comp_f1 always; the
                  gap measures value-error.

True pin / connectivity F1 is NOT computable here: the GT (data/test/*.json
`components` field) carries only {ref, value, type}, no pin-to-net wiring.
That requires parsing the .kicad_sch source (synthetic subset only) — tracked
as a separate Phase-2 task.

Usage:
    python eval_topology_v2.py --results results_v3_lora_s600_easy50.json
    python eval_topology_v2.py --results <file> --verbose   # per-sample
"""
import re
import json
import argparse
from pathlib import Path

# A reference designator: one or more letters followed by digits at line start.
# Matches R1, U2, IC10, Q3, A1. Rejects GND, VBUS (no digits), 1N4007 (starts digit).
RE_REFDES = re.compile(r'^([A-Za-z]+)(\d+)')

# Phase-1 extraction (eval_benchmark_v3.py:79): refdes with one of 7 prefixes,
# found ANYWHERE in text. Reproduced here so comp_f1 matches the paper number.
RE_REFDES_PHASE1 = re.compile(r'\b([RCDUJLQ]\d+)\b')


def extract_refdes_phase1(text):
    """Phase-1-compatible refdes extraction (7 prefixes, anywhere, uppercase)."""
    return set(RE_REFDES_PHASE1.findall(text))


def parse_components(text):
    """Parse flat netlist text into a list of (refdes, value) pairs.

    Convention: a line starting with a refdes begins a new component; subsequent
    non-refdes lines are its value, up to the next refdes or end of text.
    """
    comps = []
    cur_ref = None
    val_lines = []

    def flush():
        nonlocal cur_ref, val_lines
        if cur_ref is not None:
            comps.append((cur_ref, ' '.join(val_lines).strip()))
        cur_ref = None
        val_lines = []

    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        m = RE_REFDES.match(line)
        if m:
            flush()
            cur_ref = m.group(0)          # e.g. "R1" (letters+digits, first token)
            val_lines = []
        else:
            if cur_ref is not None:
                val_lines.append(line)
            # orphan line (e.g. a net label with no preceding refdes) is ignored
    flush()
    return comps


def norm_value(v):
    """Case-insensitive, whitespace-collapsed value comparison."""
    return re.sub(r'\s+', '', v).lower()


def metrics_for_sample(pred_text, ref_text):
    # --- comp_f1: Phase-1-compatible (anywhere regex, 7 prefixes, per-sample) ---
    pred_refs = extract_refdes_phase1(pred_text)
    ref_refs = extract_refdes_phase1(ref_text)
    if not pred_refs and not ref_refs:
        comp_f1 = comp_p = comp_r = 1.0
    elif not pred_refs or not ref_refs:
        comp_f1 = comp_p = comp_r = 0.0
    else:
        tp = len(pred_refs & ref_refs)
        comp_p = tp / len(pred_refs)
        comp_r = tp / len(ref_refs)
        comp_f1 = 2 * comp_p * comp_r / max(comp_p + comp_r, 1e-9)

    # --- value_acc / joint_f1: structure-aware (refdes, value) pairs ---
    pred_map = {r: v for r, v in parse_components(pred_text)}  # last wins
    ref_map = {r: v for r, v in parse_components(ref_text)}
    inter = set(pred_map) & set(ref_map)
    if inter:
        value_acc = sum(
            1 for ref in inter
            if norm_value(pred_map[ref]) == norm_value(ref_map[ref])
        ) / len(inter)
    else:
        value_acc = 0.0

    pred_pair_set = {(r, norm_value(v)) for r, v in pred_map.items()}
    ref_pair_set = {(r, norm_value(v)) for r, v in ref_map.items()}
    joint_inter = pred_pair_set & ref_pair_set
    if not pred_pair_set and not ref_pair_set:
        joint_f1 = 1.0
    elif not pred_pair_set or not ref_pair_set:
        joint_f1 = 0.0
    else:
        jp = len(joint_inter) / len(pred_pair_set)
        jr = len(joint_inter) / len(ref_pair_set)
        joint_f1 = 2 * jp * jr / max(jp + jr, 1e-9)

    return {
        'comp_f1': round(comp_f1, 4),
        'comp_precision': round(comp_p, 4),
        'comp_recall': round(comp_r, 4),
        'value_acc': round(value_acc, 4),
        'joint_f1': round(joint_f1, 4),
        'pred_count': len(pred_refs),
        'ref_count': len(ref_refs),
        'matched': len(pred_refs & ref_refs),
    }


def evaluate_file(results_path, verbose=False):
    with open(results_path, encoding='utf-8') as f:
        d = json.load(f)
    samples = d.get('results', [])
    agg = {k: 0.0 for k in
           ('comp_f1', 'comp_precision', 'comp_recall', 'value_acc', 'joint_f1')}
    agg['pred_count'] = agg['ref_count'] = agg['matched'] = 0
    per_sample = []
    for s in samples:
        pred = s.get('prediction', '')
        ref = s.get('reference') or s.get('ground_truth') or s.get('label', '')
        m = metrics_for_sample(pred, ref)
        per_sample.append((s.get('image', ''), m))
        # average over ALL samples (matches Phase-1 compute_component_f1)
        for k in ('comp_f1', 'comp_precision', 'comp_recall', 'value_acc', 'joint_f1'):
            agg[k] += m[k]
        agg['pred_count'] += m['pred_count']
        agg['ref_count'] += m['ref_count']
        agg['matched'] += m['matched']
    n = max(len(samples), 1)
    summary = {
        'samples': len(samples),
        'avg_comp_f1': round(agg['comp_f1'] / n, 4),
        'avg_comp_precision': round(agg['comp_precision'] / n, 4),
        'avg_comp_recall': round(agg['comp_recall'] / n, 4),
        'avg_value_acc': round(agg['value_acc'] / n, 4),
        'avg_joint_f1': round(agg['joint_f1'] / n, 4),
        'total_pred': agg['pred_count'], 'total_ref': agg['ref_count'],
        'total_matched': agg['matched'],
    }
    if verbose:
        for i, (img, m) in enumerate(per_sample):
            name = Path(img).name
            print(f"[{i+1:>2}/{len(per_sample)}] {name[:42]:<42} "
                  f"compF1={m['comp_f1']:.3f} valAcc={m['value_acc']:.3f} "
                  f"jointF1={m['joint_f1']:.3f} (pred={m['pred_count']} ref={m['ref_count']})")
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='results_v3_*.json file')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()
    s = evaluate_file(args.results, args.verbose)
    print(f"\n===== Topology v2: {Path(args.results).name} =====")
    for k, v in s.items():
        print(f"  {k}: {v}")
