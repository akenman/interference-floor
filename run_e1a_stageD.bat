@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d D:\xiangmu\世界模型
echo === Stage D: 解耦消融（容量固定 64 专家）seeds 45-51 ===
echo === A1x64 监督1轴+门控只读1轴 | A2x64 监督2轴 | A3g1 监督3轴但门控只读1轴 ===
echo === 基准为 V3（Stage A 已给出 n=10），故不跑 A3x64（与 V3 逐位等价）===
"D:\Program Files\Python314\python.exe" -u run_e1a.py --variants A1x64 A2x64 A3g1 --seeds 45 46 47 48 49 50 51 --arms normal --skip-done --log-root logs-corrected --checkpoint-root checkpoints-e1a-corrected --windows-per-epoch 60000 --w_roll 0.05 --lr 5e-4 --max-jobs 1 >> logs-corrected\scheduler_stageD.log 2>&1
echo === Stage D done ===
