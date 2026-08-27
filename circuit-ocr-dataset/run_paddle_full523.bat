@echo off
cd /d g:\mimo_project\circuit_ocr\circuit-ocr-dataset
python scripts\eval_benchmark.py --model_type paddleocr-vl --model_name_or_path PaddlePaddle/PaddleOCR-VL --data_path ocr_vl_sft-test.jsonl --output_path results_paddleocr-vl_full523.jsonl --max_length 1024 --resume 2>&1
echo EXIT_CODE=%ERRORLEVEL%
