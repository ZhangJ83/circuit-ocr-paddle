#!/usr/bin/env python3
"""
Training Script
================
Train text detection and recognition models for circuit schematic OCR.

Usage:
    python scripts/train.py --task det --distributed
    python scripts/train.py --task rec
    python scripts/train.py --task vl
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.train_det import DetTrainer
from src.model.train_rec import RecTrainer
from src.model.finetune_vl import VLFineTuner


def main():
    parser = argparse.ArgumentParser(description="Train circuit OCR models")
    parser.add_argument("--task", required=True, choices=["det", "rec", "vl", "all"],
                       help="Training task: det (detection), rec (recognition), vl (VLM), all")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--pretrained", default=None, help="Pretrained model path")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--gpu-ids", default="0", help="GPU IDs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.task in ("det", "all"):
        logger.info("Starting detection model training...")
        det_trainer = DetTrainer(
            data_dir=args.data_dir,
            output_dir=f"{args.output_dir}/det",
            pretrained_model=args.pretrained or "ch_PP-OCRv4_det_train",
        )
        det_trainer.train(
            use_distributed=args.distributed,
            gpu_ids=args.gpu_ids,
        )

    if args.task in ("rec", "all"):
        logger.info("Starting recognition model training...")
        rec_trainer = RecTrainer(
            data_dir=args.data_dir,
            output_dir=f"{args.output_dir}/rec",
            pretrained_model=args.pretrained or "ch_PP-OCRv4_rec_train",
        )
        rec_trainer.train(
            use_distributed=args.distributed,
            gpu_ids=args.gpu_ids,
        )

    if args.task in ("vl", "all"):
        logger.info("Starting VLM fine-tuning...")
        vl_tuner = VLFineTuner(
            data_dir=args.data_dir,
            output_dir=f"{args.output_dir}/vl",
            task="multi_task",
        )
        vl_tuner.train(epochs=args.epochs, batch_size=args.batch_size)

    logger.info("Training complete!")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
