#!/usr/bin/env python3
"""
Evaluation Script
==================
Evaluate the circuit schematic OCR system.

Usage:
    python scripts/evaluate.py --eval-dir data/eval --output-dir output/eval
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
from src.evaluation.evaluator import CircuitEvaluator
from src.evaluation.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Evaluate circuit OCR system")
    parser.add_argument("--eval-dir", default="data/eval", help="Evaluation data dir")
    parser.add_argument("--output-dir", default="output/eval", help="Output directory")
    parser.add_argument("--det-model", default=None, help="Detection model path")
    parser.add_argument("--rec-model", default=None, help="Recognition model path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    predictor = CircuitPredictor(
        det_model_dir=args.det_model,
        rec_model_dir=args.rec_model,
    )
    post_processor = PostProcessor()
    netlist_extractor = NetlistExtractor()

    # Load ground truths
    predictions = []
    ground_truths = []

    logger.info(f"Running evaluation on {eval_dir}...")

    for ann_file in sorted(eval_dir.glob("*.json")):
        with open(ann_file, "r", encoding="utf-8") as f:
            gt = json.load(f)

        image_path = gt.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        # Run prediction
        try:
            pred_result = predictor.predict(image_path)
            pred_result = post_processor.process(pred_result)
            netlist = netlist_extractor.extract(pred_result)

            pred_dict = pred_result.to_dict()
            pred_dict["nets"] = netlist.get("nets", {})
            predictions.append(pred_dict)
            ground_truths.append(gt)

        except Exception as e:
            logger.warning(f"Prediction failed for {image_path}: {e}")

    if not predictions:
        logger.error("No predictions generated!")
        return

    # Run evaluation
    evaluator = CircuitEvaluator(str(eval_dir))
    results = evaluator.evaluate_dataset(predictions, ground_truths)
    complexity_results = evaluator.evaluate_by_complexity(predictions, ground_truths)
    category_results = evaluator.evaluate_by_category(predictions, ground_truths)

    # Generate report
    reporter = ReportGenerator(str(output_dir))
    report_path = reporter.generate(
        results,
        model_name="CircuitOCR",
        complexity_results=complexity_results,
        category_results=category_results,
    )

    logger.info(f"\nEvaluation complete!")
    logger.info(f"Composite Score: {results['composite_score']:.2f}/100")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
