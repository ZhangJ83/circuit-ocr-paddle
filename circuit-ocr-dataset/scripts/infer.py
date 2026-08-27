#!/usr/bin/env python3
"""
Inference Script
=================
Run inference on circuit schematic images.

Usage:
    python scripts/infer.py --image path/to/schematic.png
    python scripts/infer.py --image-dir path/to/images/ --output-dir results/
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import CircuitPredictor
from src.inference.post_processor import PostProcessor
from src.inference.netlist_extractor import NetlistExtractor
from src.verification.spice_verifier import SPICEVerifier
from src.verification.auto_corrector import AutoCorrector


def main():
    parser = argparse.ArgumentParser(description="Run circuit schematic OCR inference")
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image-dir", default=None, help="Directory of images")
    parser.add_argument("--output-dir", default="output/inference", help="Output directory")
    parser.add_argument("--det-model", default=None, help="Detection model path")
    parser.add_argument("--rec-model", default=None, help="Recognition model path")
    parser.add_argument("--verify", action="store_true", help="Run SPICE verification")
    parser.add_argument("--save-netlist", action="store_true", help="Save SPICE netlist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    predictor = CircuitPredictor(
        det_model_dir=args.det_model,
        rec_model_dir=args.rec_model,
    )
    post_processor = PostProcessor()
    netlist_extractor = NetlistExtractor()
    verifier = SPICEVerifier() if args.verify else None
    corrector = AutoCorrector() if args.verify else None

    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    elif args.image_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            images.extend(Path(args.image_dir).glob(ext))
        images = [str(p) for p in images]
    else:
        parser.error("Provide --image or --image-dir")

    logger.info(f"Processing {len(images)} images...")

    for i, image_path in enumerate(images):
        try:
            logger.info(f"[{i+1}/{len(images)}] {Path(image_path).name}")

            # Step 1: OCR Prediction
            result = predictor.predict(image_path)

            # Step 2: Post-processing
            result = post_processor.process(result)

            # Step 3: Netlist extraction
            netlist = netlist_extractor.extract(result)

            # Step 4: SPICE verification (optional)
            verification = None
            if verifier and netlist.get("spice_netlist"):
                verification = verifier.verify(netlist["spice_netlist"])

                # Auto-correct if issues found
                if corrector and verification and not verification.is_valid:
                    corrected = corrector.correct(
                        netlist["spice_netlist"], verification
                    )
                    netlist["spice_netlist_corrected"] = corrected
                    verification_corrected = verifier.verify(corrected)
                    netlist["verification"] = verification_corrected.to_dict()
                elif verification:
                    netlist["verification"] = verification.to_dict()

            # Save results
            stem = Path(image_path).stem
            result_path = output_dir / f"{stem}_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(netlist, f, indent=2, ensure_ascii=False)

            # Save netlist
            if args.save_netlist and netlist.get("spice_netlist"):
                netlist_path = output_dir / f"{stem}.cir"
                netlist_path.write_text(netlist["spice_netlist"], encoding="utf-8")

            logger.info(
                f"  Texts: {len(result.texts)}, "
                f"Components: {len(netlist.get('components', []))}, "
                f"Nets: {len(netlist.get('nets', {}))}"
            )

        except Exception as e:
            logger.error(f"Failed: {e}")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
