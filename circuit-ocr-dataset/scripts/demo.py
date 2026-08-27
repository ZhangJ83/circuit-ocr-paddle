#!/usr/bin/env python3
"""
Demo Script
============
Demonstrate the circuit schematic OCR pipeline end-to-end.

Usage:
    python scripts/demo.py
    python scripts/demo.py --image path/to/schematic.png
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_synthetic():
    """Demo: Generate a synthetic circuit and run the full pipeline."""
    from src.data_pipeline.synthetic_generator import SyntheticSchematicGenerator
    from src.data_pipeline.kicad_parser import KiCadParser
    from src.data_pipeline.annotation_generator import AnnotationGenerator
    from src.data_pipeline.degradation import SchematicDegradation
    from src.inference.predictor import CircuitPredictor
    from src.inference.post_processor import PostProcessor
    from src.inference.netlist_extractor import NetlistExtractor

    logger = logging.getLogger(__name__)

    # Step 1: Generate synthetic circuit
    logger.info("Step 1: Generating synthetic circuit...")
    gen = SyntheticSchematicGenerator(output_dir="examples/demo_output")
    spec = gen.generate_random_circuit(
        num_components=10,
        circuit_type="mixed",
    )
    png_path = gen.render_synthetic_schematic(spec, "examples/demo_output/demo_circuit.png")
    logger.info(f"  Generated: {png_path}")

    # Step 2: Apply degradation
    logger.info("Step 2: Applying degradation...")
    from PIL import Image
    img = Image.open(png_path)
    degraded, degradations = SchematicDegradation.apply_random_degradation(
        img, severity_range=(0.3, 0.6), num_degradations=2
    )
    degraded_path = "examples/demo_output/demo_degraded.png"
    degraded.save(degraded_path)
    logger.info(f"  Degraded: {degradations}")

    # Step 3: Run OCR
    logger.info("Step 3: Running OCR prediction...")
    predictor = CircuitPredictor(use_paddleocr=False)
    post_processor = PostProcessor()
    netlist_extractor = NetlistExtractor()

    # Note: Without PaddleOCR, we use the annotation data directly
    logger.info("  (Demo mode: using synthetic annotations as OCR output)")

    # Step 4: Show results
    logger.info("Step 4: Results")
    logger.info(f"  Components: {len(spec['components'])}")
    logger.info(f"  Connections: {len(spec['connections'])}")
    logger.info(f"  Net labels: {len(spec['net_labels'])}")

    for comp in spec['components'][:5]:
        logger.info(f"    {comp.reference} = {comp.value} ({comp.template.name})")

    logger.info("\nDemo complete! See examples/demo_output/")


def demo_parser():
    """Demo: Parse a KiCad schematic file."""
    from src.data_pipeline.kicad_parser import KiCadParser

    parser = KiCadParser()

    # Find any .kicad_sch files
    sch_files = list(Path("data/raw").rglob("*.kicad_sch"))
    if not sch_files:
        print("No .kicad_sch files found in data/raw/")
        print("Run scripts/collect_data.py first, or use --generate flag")
        return

    sch_path = str(sch_files[0])
    print(f"Parsing: {sch_path}")

    data = parser.parse(sch_path)

    print(f"\nResults:")
    print(f"  Components: {len(data.components)}")
    for c in data.components[:5]:
        print(f"    {c.reference} = {c.value}")

    print(f"  Wires: {len(data.wires)}")
    print(f"  Labels: {len(data.labels)}")
    for l in data.labels[:5]:
        print(f"    {l.name}")

    nets = parser.extract_netlist(data)
    print(f"  Nets: {len(nets)}")
    for name, pins in list(nets.items())[:5]:
        print(f"    {name}: {pins}")


def main():
    parser = argparse.ArgumentParser(description="Circuit OCR Demo")
    parser.add_argument("--generate", action="store_true",
                       help="Run synthetic generation demo")
    parser.add_argument("--parse", action="store_true",
                       help="Run parser demo")
    parser.add_argument("--image", default=None,
                       help="Run inference demo on image")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    Path("examples/demo_output").mkdir(parents=True, exist_ok=True)

    if args.generate:
        demo_synthetic()
    elif args.parse:
        demo_parser()
    elif args.image:
        from src.inference.predictor import CircuitPredictor
        predictor = CircuitPredictor()
        result = predictor.predict(args.image)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Circuit Schematic OCR - Demo")
        print("=" * 40)
        print("\nOptions:")
        print("  --generate   Generate synthetic circuit demo")
        print("  --parse      Parse a KiCad file demo")
        print("  --image PATH Run inference on an image")
        print("\nRunning synthetic generation demo...\n")
        demo_synthetic()


if __name__ == "__main__":
    main()
