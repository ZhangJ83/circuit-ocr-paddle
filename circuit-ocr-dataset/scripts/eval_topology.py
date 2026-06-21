#!/usr/bin/env python3
"""
Topology-Aware Circuit OCR Evaluation Script
=============================================
Extends the standard NED metric with circuit-structure-aware metrics:

1. Traditional NED (Levenshtein distance, character-level)
2. Component-level F1 (REFDES matching)
3. Type Accuracy (component type/value exact match)
4. Pin Accuracy (pin connectivity match, order-independent)
5. Topology NED (weighted combination of above)

Component-list format (post-unification):
    REFDES
    TYPE/VALUE
    PIN1
    PIN2
    ...

Usage:
    python scripts/eval_topology.py \
        --results results_paddleocr-vl_easy50.jsonl \
        --output topology_report_easy50.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Try to import Levenshtein for traditional NED
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


def parse_component_list(text):
    """
    Parse component-list format text into structured components.

    Format:
        REFDES
        TYPE/VALUE
        PIN1
        PIN2
        ...

    Components are separated by blank lines or by detecting the next REFDES pattern.
    Returns: list of dicts with keys: refdes, type_val, pins
    """
    if not text or not text.strip():
        return []

    lines = [l.strip() for l in text.strip().split('\n')]
    lines = [l for l in lines if l]  # remove empty lines

    components = []
    current = None

    # REFDES patterns: R1, C2, U3, M0, Q1, L2, D3, J1, F1, X1, etc.
    # Also matches multi-char prefixes: REF1, VDD, GND, etc.
    refdes_re = re.compile(
        r'^([A-Za-z]+[A-Za-z0-9_]*\d+[A-Za-z]*|[A-Z]+)$'
    )

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line looks like a REFDES (start of new component)
        is_refdes = bool(refdes_re.match(line))
        # Also check: if line is ALL_CAPS_OR_DIGITS and not too long, likely a refdes/value
        is_short_id = (len(line) <= 20 and (
            line.isupper() or
            (line[0].isalpha() and any(c.isdigit() for c in line))
        ))

        if is_refdes:
            # Check if this is actually a REFDES or a value line
            # Heuristic: REFDES typically has both letters AND digits at end
            # e.g., "R1", "C10", "U3A" vs "10uF", "100", "GND", "VDD"
            is_likely_value = (
                line.replace('.', '').replace('-', '').replace('_', '').replace(' ', '').isdigit() or
                re.match(r'^[\d.]+[munpKkM]?[A-Za-z]*$', line) or  # 10uF, 100n, 3K, 2.2k
                line in ('VDD', 'VSS', 'GND', 'VCC', 'VIN', 'VOUT', 'VBUS', 'VREF',
                         'AGND', 'DGND', 'PGND', 'VDDIO', 'VDDA', 'VSSA',
                         'CLK', 'RST', 'EN', 'IN', 'OUT', 'NC')
            )

            if is_likely_value and current is not None:
                # This is likely a value, not a new component
                current['type_val'] = line
                i += 1
                continue

            # Start new component
            if current is not None:
                components.append(current)

            current = {'refdes': line, 'type_val': '', 'pins': []}
            i += 1

            # Next line is type/value (if not another refdes)
            if i < len(lines):
                next_line = lines[i]
                next_is_refdes = bool(refdes_re.match(next_line))
                if not next_is_refdes:
                    current['type_val'] = next_line
                    i += 1
                # else: type_val stays empty
        else:
            # This is a pin line
            if current is not None:
                current['pins'].append(line)
            else:
                # Orphan line - start a component with unknown refdes
                current = {'refdes': f'?UNKNOWN?{len(components)}', 'type_val': '', 'pins': [line]}
            i += 1

    if current is not None:
        components.append(current)

    return components


def match_components(pred_comps, ref_comps):
    """
    Match predicted components to reference components by REFDES.

    Returns:
        matched: list of (pred_comp, ref_comp) tuples
        unmatched_pred: list of pred components with no ref match
        unmatched_ref: list of ref components with no pred match
    """
    pred_by_refdes = {c['refdes']: c for c in pred_comps}
    ref_by_refdes = {c['refdes']: c for c in ref_comps}

    matched = []
    unmatched_pred = []
    unmatched_ref = []

    # Match by exact REFDES
    for refdes, ref_comp in ref_by_refdes.items():
        if refdes in pred_by_refdes:
            matched.append((pred_by_refdes[refdes], ref_comp))
        else:
            unmatched_ref.append(ref_comp)

    for refdes, pred_comp in pred_by_refdes.items():
        if refdes not in ref_by_refdes:
            unmatched_pred.append(pred_comp)

    return matched, unmatched_pred, unmatched_ref


def compute_component_metrics(pred_comps, ref_comps):
    """
    Compute component-level precision, recall, F1.
    """
    matched, unmatched_pred, unmatched_ref = match_components(pred_comps, ref_comps)

    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_ref)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'component_precision': precision,
        'component_recall': recall,
        'component_f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'matched': matched,
        'unmatched_pred': unmatched_pred,
        'unmatched_ref': unmatched_ref,
    }


def compute_type_accuracy(matched_comps):
    """
    Compute type/value exact match accuracy for matched components.
    """
    if not matched_comps:
        return 0.0, 0

    correct = 0
    for pred, ref in matched_comps:
        if pred['type_val'].strip() == ref['type_val'].strip():
            correct += 1

    return correct / len(matched_comps), correct


def compute_pin_accuracy(matched_comps):
    """
    Compute pin-level accuracy for matched components.
    Pins are compared as sets (order-independent).
    Also compute per-pin precision/recall.
    """
    if not matched_comps:
        return 0.0, 0.0, 0.0, 0

    total_pin_precision = 0.0
    total_pin_recall = 0.0
    total_pin_f1 = 0.0
    exact_pin_match = 0

    for pred, ref in matched_comps:
        pred_pins = set(pred['pins'])
        ref_pins = set(ref['pins'])

        if not pred_pins and not ref_pins:
            exact_pin_match += 1
            total_pin_precision += 1.0
            total_pin_recall += 1.0
            total_pin_f1 += 1.0
            continue

        if not pred_pins or not ref_pins:
            continue

        intersection = pred_pins & ref_pins

        pin_prec = len(intersection) / len(pred_pins)
        pin_rec = len(intersection) / len(ref_pins)
        pin_f1 = 2 * pin_prec * pin_rec / (pin_prec + pin_rec) if (pin_prec + pin_rec) > 0 else 0.0

        total_pin_precision += pin_prec
        total_pin_recall += pin_rec
        total_pin_f1 += pin_f1

        if pred_pins == ref_pins:
            exact_pin_match += 1

    n = len(matched_comps)
    return total_pin_precision / n, total_pin_recall / n, total_pin_f1 / n, exact_pin_match


def compute_topology_ned(comp_metrics, type_acc, pin_metrics, trad_ned):
    """
    Compute a topology-aware NED that combines all structural metrics.

    Formula:
        TopoNED = 0.25 * (1 - Component_F1) + 0.25 * (1 - Type_Acc)
                + 0.25 * (1 - Pin_F1) + 0.25 * Trad_NED

    Lower is better (0 = perfect match).
    """
    pin_prec, pin_rec, pin_f1, _ = pin_metrics
    comp_f1 = comp_metrics['component_f1']

    topo_ned = (
        0.25 * (1.0 - comp_f1) +
        0.25 * (1.0 - type_acc) +
        0.25 * (1.0 - pin_f1) +
        0.25 * trad_ned
    )

    return topo_ned


def compute_levenshtein_ned(pred, ref):
    """Compute traditional character-level Normalized Edit Distance."""
    if not HAS_LEVENSHTEIN:
        # Fallback: simple char-level edit distance
        return _simple_ned(pred, ref)
    dist = Levenshtein.distance(pred, ref)
    max_len = max(len(pred), len(ref))
    return dist / max_len if max_len > 0 else 0.0


def _simple_ned(s1, s2):
    """Fallback Levenshtein distance without python-Levenshtein."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(insert, delete, sub))
        prev = curr

    return prev[-1] / max(len(s1), len(s2))


def evaluate_results(results_file):
    """Evaluate all metrics on a results file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = [json.loads(l) for l in f if l.strip()]

    if not results:
        return None

    metrics = {
        'total_samples': len(results),
        'trad_ned': [],
        'component_precision': [],
        'component_recall': [],
        'component_f1': [],
        'type_accuracy': [],
        'pin_precision': [],
        'pin_recall': [],
        'pin_f1': [],
        'exact_pin_match_rate': [],
        'topology_ned': [],
        'per_sample': [],
    }

    for r in results:
        pred = r.get('prediction', '')
        ref = r.get('label', '')

        # Traditional NED
        trad_ned = compute_levenshtein_ned(pred, ref)
        metrics['trad_ned'].append(trad_ned)

        # Parse both
        pred_comps = parse_component_list(pred)
        ref_comps = parse_component_list(ref)

        # Component metrics
        comp_m = compute_component_metrics(pred_comps, ref_comps)
        metrics['component_precision'].append(comp_m['component_precision'])
        metrics['component_recall'].append(comp_m['component_recall'])
        metrics['component_f1'].append(comp_m['component_f1'])

        # Type accuracy
        type_acc, type_correct = compute_type_accuracy(comp_m['matched'])
        metrics['type_accuracy'].append(type_acc)

        # Pin accuracy
        pin_prec, pin_rec, pin_f1, exact_pins = compute_pin_accuracy(comp_m['matched'])
        metrics['pin_precision'].append(pin_prec)
        metrics['pin_recall'].append(pin_rec)
        metrics['pin_f1'].append(pin_f1)
        metrics['exact_pin_match_rate'].append(
            exact_pins / len(comp_m['matched']) if comp_m['matched'] else 0.0
        )

        # Topology NED
        topo_ned = compute_topology_ned(comp_m, type_acc, (pin_prec, pin_rec, pin_f1, exact_pins), trad_ned)
        metrics['topology_ned'].append(topo_ned)

        # Per-sample detail
        metrics['per_sample'].append({
            'image': r.get('images', ['unknown'])[0] if 'images' in r else 'unknown',
            'trad_ned': trad_ned,
            'topology_ned': topo_ned,
            'comp_f1': comp_m['component_f1'],
            'comp_precision': comp_m['component_precision'],
            'comp_recall': comp_m['component_recall'],
            'type_accuracy': type_acc,
            'pin_f1': pin_f1,
            'tp': comp_m['tp'],
            'fp': comp_m['fp'],
            'fn': comp_m['fn'],
            'pred_comps': len(pred_comps),
            'ref_comps': len(ref_comps),
        })

    return metrics


def print_report(metrics, label=""):
    """Print a formatted metrics report."""
    n = metrics['total_samples']

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'='*70}")
    print(f"  Topology-Aware Evaluation Report{f' — {label}' if label else ''}")
    print(f"{'='*70}")
    print(f"  Samples: {n}")
    print(f"{'-'*70}")
    print(f"  METRIC                          |    MEAN  |   MEDIAN |    MIN  |    MAX")
    print(f"{'-'*70}")

    metric_specs = [
        ("Traditional NED (char-level)", "trad_ned"),
        ("Component Precision", "component_precision"),
        ("Component Recall", "component_recall"),
        ("Component F1", "component_f1"),
        ("Type/Value Accuracy", "type_accuracy"),
        ("Pin Precision", "pin_precision"),
        ("Pin Recall", "pin_recall"),
        ("Pin F1", "pin_f1"),
        ("Topology NED (combined)", "topology_ned"),
    ]

    for name, key in metric_specs:
        vals = metrics[key]
        mean_v = avg(vals)
        sorted_v = sorted(vals)
        median_v = sorted_v[len(sorted_v)//2] if sorted_v else 0.0
        min_v = min(vals) if vals else 0.0
        max_v = max(vals) if vals else 0.0
        print(f"  {name:<32} | {mean_v:8.4f} | {median_v:8.4f} | {min_v:8.4f} | {max_v:8.4f}")

    print(f"{'='*70}")

    return avg(metrics['topology_ned'])


def main():
    parser = argparse.ArgumentParser(description="Topology-aware circuit OCR evaluation")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results JSONL file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for detailed per-sample report (JSON)')
    parser.add_argument('--label', type=str, default='',
                        help='Label for this evaluation run')
    parser.add_argument('--all_tiers', action='store_true',
                        help='Evaluate all tier result files (easy50, easy100, easy200)')
    args = parser.parse_args()

    if args.all_tiers:
        base_dir = Path(args.results).parent
        tiers = ['easy50', 'easy100', 'easy200']
        all_reports = {}
        for tier in tiers:
            # Try to find the result file
            patterns = [
                f'results_paddleocr-vl_{tier}.jsonl',
                f'results_qwen3-vl_{tier}.jsonl',
                f'results_qwen3-vl-lora_{tier}.jsonl',
                f'results_lora_{tier}.jsonl',
            ]
            found = None
            for p in patterns:
                candidate = base_dir / p
                if candidate.exists():
                    found = candidate
                    break

            if found:
                print(f"\n--- Evaluating {found.name} ---")
                metrics = evaluate_results(str(found))
                if metrics:
                    topo_ned = print_report(metrics, label=f"{tier} (paddleocr-vl base)")
                    all_reports[tier] = {
                        'file': str(found),
                        'avg_trad_ned': sum(metrics['trad_ned'])/len(metrics['trad_ned']),
                        'avg_topo_ned': topo_ned,
                        'avg_comp_f1': sum(metrics['component_f1'])/len(metrics['component_f1']),
                        'avg_type_acc': sum(metrics['type_accuracy'])/len(metrics['type_accuracy']),
                        'avg_pin_f1': sum(metrics['pin_f1'])/len(metrics['pin_f1']),
                    }
            else:
                print(f"  No results found for tier: {tier}")

        # Summary table
        print(f"\n{'='*70}")
        print(f"  BASELINE SUMMARY — PaddleOCR-VL (base, no fine-tuning)")
        print(f"{'='*70}")
        print(f"  {'Tier':<12} {'Trad NED':>10} {'Topo NED':>10} {'Comp F1':>10} {'Type Acc':>10} {'Pin F1':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for tier in ['easy50', 'easy100', 'easy200']:
            if tier in all_reports:
                r = all_reports[tier]
                print(f"  {tier:<12} {r['avg_trad_ned']:>10.4f} {r['avg_topo_ned']:>10.4f} {r['avg_comp_f1']:>10.4f} {r['avg_type_acc']:>10.4f} {r['avg_pin_f1']:>10.4f}")
        print(f"{'='*70}")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_reports, f, indent=2, ensure_ascii=False)
            print(f"\nSummary saved to: {args.output}")
    else:
        metrics = evaluate_results(args.results)
        if metrics is None:
            print(f"ERROR: No results found in {args.results}")
            sys.exit(1)

        print_report(metrics, label=args.label)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': {
                        'total_samples': metrics['total_samples'],
                        'avg_trad_ned': sum(metrics['trad_ned'])/len(metrics['trad_ned']),
                        'avg_topo_ned': sum(metrics['topology_ned'])/len(metrics['topology_ned']),
                        'avg_comp_f1': sum(metrics['component_f1'])/len(metrics['component_f1']),
                        'avg_type_acc': sum(metrics['type_accuracy'])/len(metrics['type_accuracy']),
                        'avg_pin_f1': sum(metrics['pin_f1'])/len(metrics['pin_f1']),
                    },
                    'per_sample': metrics['per_sample'],
                }, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed report saved to: {args.output}")


if __name__ == '__main__':
    main()
