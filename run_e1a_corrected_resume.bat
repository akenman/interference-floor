@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d D:\xiangmu\世界模型
"D:\Program Files\Python314\python.exe" -u run_e1a.py --log-root logs-corrected --checkpoint-root checkpoints-e1a-corrected --windows-per-epoch 60000 --w_roll 0.05 --lr 5e-4 --max-jobs 1 --resume --skip-done >> logs-corrected\scheduler.log 2>&1
