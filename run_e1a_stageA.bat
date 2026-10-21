@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d D:\xiangmu\世界模型
echo === Stage A: 常态臂扩种子 45-51（V1 V2a V2b V3 Vfree）===
"D:\Program Files\Python314\python.exe" -u run_e1a.py --variants V1 V2a V2b V3 Vfree --seeds 45 46 47 48 49 50 51 --arms normal --skip-done --log-root logs-corrected --checkpoint-root checkpoints-e1a-corrected --windows-per-epoch 60000 --w_roll 0.05 --lr 5e-4 --max-jobs 1 >> logs-corrected\scheduler_stageA.log 2>&1
echo === Stage A done ===
