@echo off
cd /d g:\mimo_project\circuit_ocr\circuit-ocr-dataset
E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\python.exe scripts\eval_benchmark.py --model_type qwen3-vl --model_name_or_path Qwen/Qwen3-VL-8B-Instruct --data_path ocr_vl_sft-test-easy200.jsonl --output_path results_qwen3-vl_easy200.jsonl --max_length 1024 --resume 2>&1
echo EXIT_CODE=%ERRORLEVEL%
