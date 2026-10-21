@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d D:\xiangmu\世界模型
echo === Stage 1: V2a s43/s44 ===
"D:\Program Files\Python314\python.exe" -u run_e1a.py --variants V2a --seeds 43 44 --w_roll 0.05 --lr 5e-4 --max-jobs 1 >> logs\e1a_resume.log 2>&1
echo === Stage 2: V2b/V2c/V3/V4/Vfree all seeds ===
"D:\Program Files\Python314\python.exe" -u run_e1a.py --variants V2b V2c V3 V4 Vfree --seeds 42 43 44 --w_roll 0.05 --lr 5e-4 --max-jobs 1 >> logs\e1a_resume.log 2>&1
echo === All done ===
