"""Evaluation metrics for circuit OCR. Extracted from eval_benchmark_v3.py."""
import re
from collections import defaultdict

# ── Component extraction ──
_COMP_PATTERN = re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')

def extract_components(text):
    """Extract component refdes like R1, C2, U3, J4, D5, L6, Q7."""
    return _COMP_PATTERN.findall(text)


# ── Value normalization ──
def normalize_value(v):
    """Normalize component values for comparison. 10k==10kohm, 100nF==0.1uF, etc."""
    v = v.strip().lower()
    v = v.replace('ohm', '').replace('Ω', '').replace('ω', '')
    v = v.replace('μ', 'u').replace('uf', 'u').replace('µf', 'u')
    v = v.replace('nf', 'n').replace('pf', 'p')
    v = v.replace(' ', '').replace('%', '')
    return v


# ── Single metrics ──
def compute_exact_match(predictions, references):
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions) if predictions else 0.0


def compute_component_f1(predictions, references):
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_comps = set(extract_components(pred))
        ref_comps = set(extract_components(ref))
        if not pred_comps and not ref_comps:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0)
        elif not pred_comps or not ref_comps:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
        else:
            tp = len(pred_comps & ref_comps)
            prec = tp / len(pred_comps) if pred_comps else 0.0
            rec = tp / len(ref_comps) if ref_comps else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
    }


def compute_token_recall(predictions, references):
    recalls = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        if not ref_tokens: recalls.append(1.0)
        elif not pred_tokens: recalls.append(0.0)
        else: recalls.append(len(pred_tokens & ref_tokens) / len(ref_tokens))
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_repetition_rate(predictions, min_repeat=4):
    repeated = 0
    for pred in predictions:
        lines = pred.strip().split('\n')
        if len(lines) < min_repeat: continue
        max_run = 1; current_run = 1
        for i in range(1, len(lines)):
            if lines[i].strip() == lines[i-1].strip():
                current_run += 1
                max_run = max(max_run, current_run)
            else: current_run = 1
        if max_run >= min_repeat: repeated += 1
    return repeated / len(predictions) if predictions else 0.0, repeated


def compute_ned(predictions, references):
    try:
        import Levenshtein
        distances = []
        for pred, ref in zip(predictions, references):
            d = Levenshtein.distance(pred, ref)
            max_len = max(len(pred), len(ref), 1)
            distances.append(d / max_len)
        return sum(distances) / len(distances) if distances else 1.0
    except ImportError:
        import difflib
        distances = []
        for pred, ref in zip(predictions, references):
            d = sum(1 for _ in difflib.ndiff(pred, ref) if _[0] != ' ')
            max_len = max(len(pred), len(ref), 1)
            distances.append(d / max_len)
        return sum(distances) / len(distances) if distances else 1.0


def compute_diversity(predictions):
    unique = len(set(predictions))
    return unique / len(predictions) if predictions else 0.0, unique


# ── Joint F1 (refdes + value pairs) ──
def compute_joint_f1(predictions, references):
    """Parse flat text into (refdes, value) pairs, normalize, compute joint F1."""
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_pairs = _parse_pairs(pred)
        ref_pairs = _parse_pairs(ref)
        if not pred_pairs and not ref_pairs:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0)
        elif not pred_pairs or not ref_pairs:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
        else:
            tp = sum(1 for k in pred_pairs if k in ref_pairs and normalize_value(pred_pairs[k]) == normalize_value(ref_pairs[k]))
            prec = tp / len(pred_pairs) if pred_pairs else 0.0
            rec = tp / len(ref_pairs) if ref_pairs else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
    }


def _parse_pairs(text):
    """Parse flat text into {refdes: value} dict.
    Heuristic: refdes line (R1) followed by value line (10k), then pins below."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    pairs = {}
    i = 0
    while i < len(lines):
        comps = extract_components(lines[i])
        if comps and len(comps) == 1:
            refdes = comps[0]
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Next line is value if it's not a refdes and not a pin (number+space)
                if not _COMP_PATTERN.match(next_line) and not re.match(r'^\d+\s', next_line):
                    pairs[refdes] = next_line
                else:
                    pairs[refdes] = ''  # no visible value
        i += 1
    return pairs


# ── All metrics ──
def compute_all(predictions, references, label=""):
    return {
        "label": label,
        "n_samples": len(predictions),
        "exact_match": round(compute_exact_match(predictions, references), 4),
        "component_f1": round(compute_component_f1(predictions, references)["f1"], 4),
        "comp_precision": round(compute_component_f1(predictions, references)["precision"], 4),
        "comp_recall": round(compute_component_f1(predictions, references)["recall"], 4),
        "token_recall": round(compute_token_recall(predictions, references), 4),
        "repetition_rate": round(compute_repetition_rate(predictions)[0], 4),
        "ned": round(compute_ned(predictions, references), 4),
        "diversity": round(compute_diversity(predictions)[0], 4),
        "joint_f1": round(compute_joint_f1(predictions, references)["f1"], 4),
    }
