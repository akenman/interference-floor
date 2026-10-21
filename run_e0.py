"""并行调度 E0 训练：每个 arm×seed 一个独立进程，共享单张 GPU。
用法：python run_e0.py [--arms a b c] [--seeds 42 43 44] [--prior_spec SPEC] [--max-jobs N]

P0-a 先验阶梯示例：
  python run_e0.py --arms ours_base --seeds 42 43 44 --prior_spec rk4001x5

P0-c 辅助监督 2×2 示例（四格只差 --checkpoint_root 与 --extra_train_args）：
  python run_e0.py --arms ours_base --seeds 42 43 44 45 46 47 --prior_spec euler005 ^
    --checkpoint_root checkpoints-P0c/C00 --w_roll 0 --lr 1e-3 --windows_per_epoch 150000 ^
    --epochs 60 --extra_train_args --w_yao 0 --w_proto 0 --w_commit 0 --w_usage 0 --max-jobs 1
"""
import argparse, subprocess, sys, os, time, signal
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_e0
from physics_prior_spec import DEFAULT_SPEC, SPECS

ARMS = list(model_e0.ARM_SPECS)
SEEDS = [42, 43, 44]


def task_name(arm, seed, prior_spec=None):
    """默认档位沿用历史命名 `<arm>_s<seed>`；非默认档位加 `@<spec>`，
    与 build_e0_report.parse_task 的约定一致。"""
    if prior_spec and prior_spec != DEFAULT_SPEC:
        return f"{arm}@{prior_spec}_s{seed}"
    return f"{arm}_s{seed}"


# E0 历史配方的坐标（root, w_roll, lr, wpe, detach, epochs）——命中则日志名不带后缀
DEFAULT_RECIPE = ("checkpoints-E0", 0.0, 1e-3, 0, False, 60)


def log_name(arm, seed, prior_spec=None, checkpoint_root="checkpoints-E0",
             w_roll=0.0, lr=1e-3, windows_per_epoch=0, detach_prior=False, epochs=60):
    """训练日志名（不含 logs/ 目录与 .log 扩展名），与 run_e0.cmd 共用同一配方语义。

    默认 E0 配方沿用历史名 `<arm>_s<seed>`；其余配方追加
    `~<root 末级名>_wr.._lr.._wpe..[_det]_ep..`。
    根目录末级名必须入名（T1，2026-10-02）：P0-c 四格（C00/C01/C10/C11）
    只差 --checkpoint_root，w_roll/lr/wpe/epochs 全同——名里不含它，四格
    会写进同一条日志互相覆盖，事后无法归因。
    """
    name = task_name(arm, seed, prior_spec)
    recipe = (checkpoint_root, w_roll, lr, windows_per_epoch,
              bool(detach_prior), epochs)
    if recipe != DEFAULT_RECIPE:
        name += (f"~{Path(checkpoint_root).name}_wr{w_roll:g}_lr{lr:g}"
                 f"_wpe{windows_per_epoch}{'_det' if detach_prior else ''}"
                 f"_ep{epochs}")
    return name


def normalize_extra_args(extra):
    """统一 --extra_train_args 的两种形态（实测 2026-10-02）：

    · 分 token 形式 `--extra_train_args --w_yao 0 …` 会被 argparse 截断——
      未知旗标（--w_yao 等）被视为选项，触发 "unrecognized arguments"；
    · 整串形式 `--extra_train_args="--w_yao 0 --w_proto 0 …"` 收成**单个**
      含空格的元素。
    本函数把两种都规整成干净的 token 列表，P0-c 四格命令两种写法等价。
    """
    out = []
    for item in extra:
        out.extend(str(item).split())
    return out


def cmd(arm, seed, prior_spec=DEFAULT_SPEC, checkpoint_root="checkpoints-E0",
        w_roll=0.0, lr=1e-3, windows_per_epoch=0, detach_prior=False, epochs=60,
        extra_train_args=()):
    spec = __import__("model_e0").ARM_SPECS
    if not spec[arm]["trainable"]:
        return None
    c = [sys.executable, "train_e0.py", "--arm", arm, "--seed", str(seed),
         "--prior_spec", prior_spec, "--epochs", str(epochs),
         "--w_roll", str(w_roll), "--lr", str(lr),
         "--windows_per_epoch", str(windows_per_epoch),
         "--checkpoint_dir",
         f"{checkpoint_root}/{task_name(arm, seed, prior_spec)}"]
    if detach_prior:
        c.append("--detach_prior")
    c.extend(extra_train_args)          # 例如 --w_yao 0 --w_proto 0（P0-c 的格子定义）
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--prior_spec", type=str, default=DEFAULT_SPEC,
                    choices=sorted(SPECS), help="先验积分器档位（P0-a 阶梯）")
    ap.add_argument("--checkpoint_root", type=str, default="checkpoints-E0",
                    help="检查点根目录；不同训练配方必须分开存放"
                         "（覆盖保护会比较 arm/seed/prior_spec/w_roll/lr）")
    ap.add_argument("--w_roll", type=float, default=0.0,
                    help="闭环 rollout 损失权重（0=纯单步；P0-b 用 0.05）")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="学习率（P0-b 的闭环配方用 5e-4，见 E1a 超参消融）")
    ap.add_argument("--windows_per_epoch", type=int, default=0,
                    help="每 epoch 训练窗口数（0=全量 507,600）。P0-b 用 150000 控成本，"
                         "两种配方必须同值")
    ap.add_argument("--detach_prior", action="store_true",
                    help="闭环 rollout 里 detach 物理先验的输入（数值不变，省约 2× 时间）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--extra_train_args", nargs="*", default=[],
                    help="追加给 train_e0.py 的参数（P0-c 的格子定义）。推荐整串形式："
                         "--extra_train_args=\"--w_yao 0 --w_proto 0\"——分 token 形式"
                         "会被 argparse 的未知旗标截断（normalize_extra_args 兼容两者）")
    ap.add_argument("--max-jobs", type=int, default=2,
                    help="并发上限；单卡 8GB 建议 2，避免 OOM")
    args = ap.parse_args()
    # 先规整再打印/下发：整串形式（--extra_train_args="--w_yao 0 …"）在
    # argparse 里是单个含空格元素，必须拆开才能透传给 train_e0
    args.extra_train_args = normalize_extra_args(args.extra_train_args)

    child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def _cmd(a, s):
        return cmd(a, s, args.prior_spec, args.checkpoint_root, args.w_roll, args.lr,
                   args.windows_per_epoch, args.detach_prior, args.epochs,
                   args.extra_train_args)

    tasks = [(a, s) for a in args.arms for s in args.seeds if _cmd(a, s) is not None]
    print(f"共 {len(tasks)} 个训练任务（prior_spec={args.prior_spec}, "
          f"w_roll={args.w_roll}, lr={args.lr}, wpe={args.windows_per_epoch}, "
          f"detach_prior={args.detach_prior}, epochs={args.epochs}, "
          f"extra={args.extra_train_args}, "
          f"root={args.checkpoint_root}），并发上限 {args.max_jobs}", flush=True)

    running = []
    done = []
    idx = 0

    while idx < len(tasks) or running:
        # 收割已结束进程
        still = []
        for p in running:
            rc = p.poll()
            if rc is None:
                still.append(p)
            else:
                done.append((p.name, rc))
        running = still

        # 启动新任务
        while len(running) < args.max_jobs and idx < len(tasks):
            arm, seed = tasks[idx]
            # 日志名带上配方标识（含 root 末级名，见 log_name）：不同配方不得共用
            # 一条日志，否则"这次 run 用的是哪个配方"会变成靠时间戳猜（P0-b 起启用；
            # T1 起补入 root——P0-c 四格只差 root）。
            name = log_name(arm, seed, args.prior_spec, args.checkpoint_root,
                            args.w_roll, args.lr, args.windows_per_epoch,
                            args.detach_prior, args.epochs)
            log = os.path.join(ROOT, f"logs/{name}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            # 追加而非截断：日志是训练期指标（best val、崩溃取证）的唯一留存，
            # 重跑同名 run 时旧记录不得被抹掉（2026-10-01 事故中已吃过一次亏）。
            fh = open(log, "a", encoding="utf-8")
            fh.write(f"\n{'='*78}\n"
                     f"[invocation] {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                     f"arm={arm} seed={seed} prior_spec={args.prior_spec} "
                     f"w_roll={args.w_roll} lr={args.lr} wpe={args.windows_per_epoch} "
                     f"detach_prior={args.detach_prior} epochs={args.epochs} "
                     f"root={args.checkpoint_root}\n{'='*78}\n")
            fh.flush()
            p = subprocess.Popen(_cmd(arm, seed), stdout=fh, stderr=subprocess.STDOUT,
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