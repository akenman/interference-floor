@echo off
REM U3：T8/M-D v2 复核——target bug 影响量化（2026-09-21 制定，2026-09-22 修复编码问题）
REM 用 %~dp0 引用脚本所在目录，避免中文路径在非 UTF-8 代码页下被 cmd 解析错乱
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM 臂1：旧行为（不把 select_scene 靶点对齐到 env，复现 M-D v2 原始条件）
"D:\Program Files\Python314\python.exe" -u eval_planning.py --episodes 100 --legacy_target --model_ckpt checkpoints-v3-calib2/best.pt --output_report analysis_output/md_planning_legacy_report.json > logs\u3_legacy.log 2>&1
REM 臂2：修复行为（env.target = select_scene 靶点）
"D:\Program Files\Python314\python.exe" -u eval_planning.py --episodes 100 --model_ckpt checkpoints-v3-calib2/best.pt --output_report analysis_output/md_planning_fixed_report.json > logs\u3_fixed.log 2>&1
echo U3 DONE > logs\u3_done.txt
