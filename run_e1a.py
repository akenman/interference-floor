#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1a 变体训练调度：爻轴组合变体 × 种子。

变体（train_e1a.py VARIANTS）：
  历史 7 变体（爻监督轴数 × 门控容量共变）：
   V1   仅初爻（接触轴）          V2b  初爻 + 上爻
   V2a  初爻 + 中爻              V2c  中爻 + 上爻
   V3   三爻全（ours-base）       V4   三爻 + 能量轴
   Vfree 三爻全 + 自由 MoE 门控

  解耦 4 变体（固定 64 专家容量，只变监督轴 / 门控可见轴）：
   A1x64 监督 1 轴 + 门控只读初爻     A2x64 监督 2 轴 + 门控只读该 2 轴
   A3g1  监督 3 轴 + 门控只读初爻     A3x64 = V3 语义（回归校验）
"""
import argparse, subprocess, sys, os, time

ROOT = os.path.dirname(os.path.abspath(__file__))

VARIANTS = ["V1", "V2a", "V2b", "V2c", "V3", "V4", "Vfree",
            "A1x64", "A2x64", "A3g1", "A3x64"]
SEEDS = [42, 43, 44]


def cmd(variant, seed, no_gua=False, w_roll=0.0, lr=1e-3, resume=False,
        checkpoint_root="checkpoints-e1a", epochs=60, windows_per_epoch=0,
        early_stop_patience=8):
    name = f"{variant}_s{seed}" + ("_nogua" if no_gua else "")
    ckpt_dir = os.path.join(checkpoint_root, name)
    c = [sys.executable, "-u", "train_e1a.py", "--variant", variant, "--seed", str(seed),
         "--no_amp", "--epochs", str(epochs),
         "--windows_per_epoch", str(windows_per_epoch),
         "--early_stop_patience", str(early_stop_patience),
         "--w_roll", str(w_roll), "--lr", str(lr),
         "--checkpoint_dir", ckpt_dir]
    latest = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    if resume and os.path.exists(latest):
        c += ["--resume", latest]      # 断点续训（2026-09-21 起 checkpoint 含 optimizer 状态）
    if no_gua:
        c.append("--no_gua")
    return c, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--max-jobs", type=int, default=1,
                    help="E0 已占 2 槽，此处 1 即可（8GB 卡）")
    ap.add_argument("--checkpoint-root", default="checkpoints-e1a")
    ap.add_argument("--log-root", default="logs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--windows-per-epoch", type=int, default=0)
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--w_roll", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true",
                    help="对已有 checkpoint_latest.pt 的任务从断点续训（暂停/恢复用）")
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过日志已含 [OK] 的任务（断点续跑调度用）")
    ap.add_argument("--arms", choices=["all", "normal", "nogua"],
                    default="all",
                    help="只跑某一类臂（默认 all；扩种子时先跑 normal）")
    args = ap.parse_args()

    # BUG-E1a-3 修复：补 no_gua 对照臂（M3 门控增益判据的合法前提）。
    # Vfree 的 no_gua 与 V3 的 no_gua 同为"无门控主干"，重复无信息，跳过。
    tasks = []
    if args.arms in ("all", "normal"):
        tasks += [(v, s, False) for v in args.variants for s in args.seeds]
    if args.arms in ("all", "nogua"):
        tasks += [(v, s, True) for v in args.variants if v != "Vfree"
                  for s in args.seeds]

    if args.skip_done:
        kept, skipped = [], 0
        for v, s, ng in tasks:
            name = f"{v}_s{s}" + ("_nogua" if ng else "")
            log = os.path.join(ROOT, args.log_root, f"{name}.log")
            done_ok = False
            if os.path.exists(log):
                try:
                    with open(log, encoding="utf-8", errors="ignore") as fh:
                        done_ok = "[OK]" in fh.read()[-4096:]
                except OSError:
                    pass
            if done_ok:
                skipped += 1
            else:
                kept.append((v, s, ng))
        if skipped:
            print(f"跳过已完成 {skipped} 个任务（--skip-done）", flush=True)
        tasks = kept
        if not tasks:
            print("无待跑任务", flush=True)
            return

    child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print(f"共 {len(tasks)} 个训练任务，并发上限 {args.max_jobs}", flush=True)

    running, done, idx = [], [], 0
    while idx < len(tasks) or running:
        still = []
        for p in running:
            rc = p.poll()
            if rc is None:
                still.append(p)
            else:
                done.append((p.name, rc))
        running = still

        while len(running) < args.max_jobs and idx < len(tasks):
            variant, seed, no_gua = tasks[idx]
            c, name = cmd(variant, seed, no_gua, args.w_roll, args.lr, args.resume,
                          args.checkpoint_root, args.epochs, args.windows_per_epoch,
                          args.early_stop_patience)
            log = os.path.join(ROOT, args.log_root, f"{name}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            # 续训时不要静默截断历史日志：先备份，供审计崩溃/收敛轨迹。
            # （直接 append 会把崩溃前后两段轨迹混在一个文件里，解析易误判。）
            if args.resume and os.path.exists(log) and os.path.getsize(log) > 0:
                bak = f"{log}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
                os.replace(log, bak)
                print(f"  [resume] 原日志已备份 -> {os.path.basename(bak)}", flush=True)
            fh = open(log, "w", encoding="utf-8", buffering=1)
            p = subprocess.Popen(c, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=ROOT, env=child_env,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            p.name = name; p.log = fh
            running.append(p)
            print(f"  [{time.strftime('%H:%M')}] 启动 {name} (pid {p.pid})", flush=True)
            idx += 1

        if not running:
            break
        time.sleep(10)

    for p in running:
        p.wait()
        done.append((p.name, p.returncode))
        p.log.close()

    print("\n结果:")
    for name, rc in done:
        print(f"  [{'OK ' if rc == 0 else 'FAIL'}] {name} (exit {rc})")
    fails = [n for n, rc in done if rc != 0]
    if fails:
        print(f"\n失败: {fails}")
        sys.exit(1)
    print("\n全部完成")


if __name__ == "__main__":
    main()
