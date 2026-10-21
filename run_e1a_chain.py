#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1a 串行链式监督：等 Stage A 收尾 → 校验 → 启动 Stage D。

用法: python run_e1a_chain.py --stage-a-pid 29108

设计要点:
  * 等 Stage A 调度器进程退出（不靠日志猜测，避免半途误判）。
  * 退出后逐一校验 35 个计划 run 的日志含 [OK]；有任何缺失/失败就**不启动**
    Stage D，直接落盘并退出非零 —— 防止上游失败静默级联。
  * 校验通过后以 detached 方式拉起 Stage D，父进程立即退出，GPU 不空转。
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(ROOT, "logs-corrected")
SCHED_A = os.path.join(LOG_ROOT, "scheduler_stageA.log")
SCHED_D = os.path.join(LOG_ROOT, "scheduler_stageD.log")
STATUS = os.path.join(LOG_ROOT, "chain_status.log")

STAGE_A_TASKS = [(v, s) for v in ["V1", "V2a", "V2b", "V3", "Vfree"]
                 for s in range(45, 52)]
STAGE_D_VARIANTS = ["A1x64", "A2x64", "A3g1"]
STAGE_D_SEEDS = list(range(45, 52))


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, errors="ignore")
    return str(pid) in (out.stdout or "")


def run_ok(variant, seed):
    log = os.path.join(LOG_ROOT, f"{variant}_s{seed}.log")
    if not os.path.exists(log):
        return False
    try:
        with open(log, encoding="utf-8", errors="ignore") as fh:
            return "[OK]" in fh.read()[-4096:]
    except OSError:
        return False


def wait_for_stage_a(pid, poll=60):
    say(f"等待 Stage A 结束 (pid={pid}) ...")
    while pid_alive(pid):
        time.sleep(poll)
    # 给文件 flush 留余量
    time.sleep(20)
    say("Stage A 进程已退出。")


def verify_stage_a():
    missing = [f"{v}_s{s}" for v, s in STAGE_A_TASKS if not run_ok(v, s)]
    if missing:
        say(f"[FAIL] Stage A 有 {len(missing)}/{len(STAGE_A_TASKS)} 个 run 未成功: {missing}")
        say("[FAIL] 不启动 Stage D（避免上游失败级联）。")
        return False
    say(f"[OK] Stage A 全部 {len(STAGE_A_TASKS)} runs 均含 [OK]。")
    return True


def launch_stage_d():
    cmd = [sys.executable, "-u", "run_e1a.py",
           "--variants", *STAGE_D_VARIANTS,
           "--seeds", *[str(s) for s in STAGE_D_SEEDS],
           "--arms", "normal", "--skip-done",
           "--log-root", "logs-corrected",
           "--checkpoint-root", "checkpoints-e1a-corrected",
           "--windows-per-epoch", "60000",
           "--w_roll", "0.05", "--lr", "5e-4", "--max-jobs", "1"]
    with open(SCHED_D, "a", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    say(f"[OK] Stage D 已启动 (pid={p.pid})，"
        f"{len(STAGE_D_VARIANTS) * len(STAGE_D_SEEDS)} 个 run，"
        f"日志 -> {os.path.relpath(SCHED_D, ROOT)}")
    return p.pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-pid", type=int, required=True)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    say("=== E1a 链式监督启动 ===")
    if not pid_alive(args.stage_a_pid):
        say(f"Stage A (pid={args.stage_a_pid}) 不在运行，直接进入校验。")
    else:
        wait_for_stage_a(args.stage_a_pid, args.poll)

    if not verify_stage_a():
        sys.exit(1)

    # 阶段 A 报告重建（用刚完成的完整数据）
    rep = subprocess.run(
        [sys.executable, "build_e1a_report.py", "--logs", "logs-corrected",
         "--out", "analysis_output/E1a_corrected_report.json",
         "--plan-normal", "42-51", "--plan-nogua", "42-44"],
        cwd=ROOT, capture_output=True, text=True, errors="ignore")
    say(f"[report] Stage A 重建完成 (exit={rep.returncode})")

    launch_stage_d()
    say("=== 链式监督结束，Stage D 独立运行中 ===")


if __name__ == "__main__":
    main()
