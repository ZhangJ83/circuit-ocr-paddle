"""Batch render .kicad_sch → SVG → PNG using kicad-cli + cairosvg.
KiCad 10 CLI exports SVG, cairosvg converts to PNG (using KiCad's Cairo DLL).
"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path

KICAD_CLI = r"E:/080000software/Kicad/bin/kicad-cli.exe"
KICAD_BIN = r"E:/080000software/Kicad/bin"

def setup_env():
    """Add KiCad bin to PATH for Cairo DLL access."""
    os.environ["PATH"] = KICAD_BIN + ";" + os.environ.get("PATH", "")
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(KICAD_BIN)
        except Exception:
            pass


def render_svg(sch_path, svg_dir):
    """Render .kicad_sch to SVG using kicad-cli."""
    os.makedirs(svg_dir, exist_ok=True)
    result = subprocess.run(
        [KICAD_CLI, "sch", "export", "svg",
         "--output", svg_dir,
         "--exclude-drawing-sheet",
         "--no-background-color",
         sch_path],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"  SVG FAIL: {os.path.basename(sch_path)}: {result.stderr.strip()}", flush=True)
        return None
    sch_name = os.path.splitext(os.path.basename(sch_path))[0]
    svg_path = os.path.join(svg_dir, f"{sch_name}.svg")
    return svg_path if os.path.exists(svg_path) else None


def svg_to_png(svg_path, png_path, dpi=150):
    """Convert SVG to PNG using cairosvg."""
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    try:
        import cairosvg
        result = cairosvg.svg2png(url=svg_path, dpi=dpi, background_color='white')
        with open(png_path, 'wb') as f:
            f.write(result)
        return True
    except Exception as e:
        print(f"  PNG FAIL: {os.path.basename(svg_path)}: {e}", flush=True)
        return False


def batch_render(sch_dir, out_dir, start_idx=0, end_idx=None):
    """Batch render all .kicad_sch files."""
    setup_env()

    svg_dir = os.path.join(out_dir, 'svg')
    png_dir = os.path.join(out_dir, 'images')
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Find all .kicad_sch files
    sch_files = sorted([f for f in os.listdir(sch_dir) if f.endswith('.kicad_sch')])
    if end_idx is not None:
        sch_files = sch_files[start_idx:end_idx]
    else:
        sch_files = sch_files[start_idx:]

    n = len(sch_files)
    if n == 0:
        print("No .kicad_sch files found!", flush=True)
        return

    print(f"Rendering {n} schematics...", flush=True)
    t0 = time.time()
    log_every = max(1, n // 20)
    success = 0

    for i, sch_name in enumerate(sch_files):
        sch_path = os.path.join(sch_dir, sch_name)
        base = os.path.splitext(sch_name)[0]
        png_path = os.path.join(png_dir, f"{base}.png")
        svg_path = os.path.join(svg_dir, f"{base}.svg")

        # Skip if PNG already exists
        if os.path.exists(png_path) and os.path.getsize(png_path) > 1000:
            success += 1
            if (i + 1) % log_every == 0:
                print(f"  [SKIP] {i+1}/{n} (already done)", flush=True)
            continue

        # Step 1: SVG export
        if not os.path.exists(svg_path):
            svg_path = render_svg(sch_path, svg_dir)
            if svg_path is None:
                print(f"  [FAIL:{sch_name}] SVG export error", flush=True)
                continue

        # Step 2: SVG → PNG
        if svg_to_png(svg_path, png_path, dpi=150):
            success += 1

        if (i + 1) % log_every == 0:
            et = (time.time() - t0) / 60
            rate = (i + 1) / max(et, 0.01)
            eta = (n - i - 1) / max(rate, 0.01)
            print(f"[RENDER] {i+1}/{n} ({rate:.0f}/min) ETA={eta:.0f}m succ={success}", flush=True)

    tt = (time.time() - t0) / 60
    print(f"\n[RENDER] Done: {success}/{n} rendered in {tt:.0f}m ({n/tt:.0f}/min)", flush=True)
    return success


def fix_jsonl(jsonl_path, png_dir):
    """Update JSONL image paths to point to actual PNG files."""
    with open(jsonl_path, encoding='utf-8') as f:
        entries = [json.loads(l) for l in f if l.strip()]

    fixed = 0
    for e in entries:
        sch_file = e.get('sch_file', '')
        base = os.path.splitext(os.path.basename(sch_file))[0]
        png_path = os.path.join(png_dir, f"{base}.png").replace('\\', '/')
        if os.path.exists(png_path):
            e['images'] = [png_path]
            fixed += 1

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f"[FIX] Updated {fixed}/{len(entries)} image paths in {jsonl_path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sch_dir', default='g:/mimo_project/circuit_ocr/output/synthetic_kicad_v3/schematics')
    ap.add_argument('--out_dir', default='g:/mimo_project/circuit_ocr/output/synthetic_kicad_v3')
    ap.add_argument('--jsonl', default='g:/mimo_project/circuit_ocr/output/synthetic_kicad_v3/train_synthetic_kicad.jsonl')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--fix_only', action='store_true')
    args = ap.parse_args()

    if args.fix_only:
        fix_jsonl(args.jsonl, os.path.join(args.out_dir, 'images'))
    else:
        n = batch_render(args.sch_dir, args.out_dir, args.start, args.end)
        if n > 0:
            fix_jsonl(args.jsonl, os.path.join(args.out_dir, 'images'))
