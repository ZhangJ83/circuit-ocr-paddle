@echo off
set PYTHON=E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe
set SCRIPT=g:\mimo_project\circuit_ocr\circuit-ocr-dataset\scripts\train_robust.py
set LOGS=g:\mimo_project\circuit_ocr\experiment_logs
mkdir %LOGS% 2>nul

echo ============================================ > %LOGS%\pipeline.log
echo PIPELINE START: %date% %time% >> %LOGS%\pipeline.log

echo [%time%] Exp1: Baseline (dim=384, lr=2e-5, epochs=2) >> %LOGS%\pipeline.log
%PYTHON% %SCRIPT% --name exp1_baseline --max_dim 384 --lr 2e-5 --epochs 2 --dropout 0.05 > %LOGS%\exp1.log 2>&1
echo [%time%] Exp1 done >> %LOGS%\pipeline.log

echo [%time%] Exp2: High-Res (dim=512, lr=2e-5, epochs=2) >> %LOGS%\pipeline.log
%PYTHON% %SCRIPT% --name exp2_hires --max_dim 512 --lr 2e-5 --epochs 2 --dropout 0.05 > %LOGS%\exp2.log 2>&1
echo [%time%] Exp2 done >> %LOGS%\pipeline.log

echo [%time%] Exp3: Anti-Overfit (dim=384, lr=1e-5, epochs=3, dropout=0.1) >> %LOGS%\pipeline.log
%PYTHON% %SCRIPT% --name exp3_regularized --max_dim 384 --lr 1e-5 --epochs 3 --dropout 0.10 > %LOGS%\exp3.log 2>&1
echo [%time%] Exp3 done >> %LOGS%\pipeline.log

echo [%time%] Exp4: Unfrozen Projector (dim=384, lr=2e-5, epochs=2) >> %LOGS%\pipeline.log
%PYTHON% %SCRIPT% --name exp4_unfrozen --max_dim 384 --lr 2e-5 --epochs 2 --freeze_projector 0 > %LOGS%\exp4.log 2>&1
echo [%time%] Exp4 done >> %LOGS%\pipeline.log

echo ============================================ >> %LOGS%\pipeline.log
echo PIPELINE DONE: %date% %time% >> %LOGS%\pipeline.log
