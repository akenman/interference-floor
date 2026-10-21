@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d D:\xiangmu\世界模型
"D:\Program Files\Python314\python.exe" -u run_e1a.py --variants V1 V2a V2b V2c V3 V4 Vfree --seeds 42 43 44 --w_roll 0.05 --lr 5e-4 --max-jobs 1 > logs\e1a_full_retrain.log 2>&1
