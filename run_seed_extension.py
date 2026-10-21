# -*- coding: utf-8 -*-
"""Seed-extension training driver: seeds 50 and 51 across the four prior tiers.

Runs up to MAX_CONCURRENT training processes, each writing its own log file.
A job counts as complete only when its log contains "Epoch 60/60" and best.pt
exists; otherwise it is reported as a failure and NOT silently retried.

Resumable: already-complete jobs are skipped on a rerun.

Usage: python -X utf8 run_seed_extension.py [--seeds 50 51] [--max-concurrent 4]
Output: analysis_output/seed_extension_status.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRA = os.path.join(ROOT, "checkpoints-E0-extra")
TIERS = [
    ("euler005", "ours_base"),
    ("euler001x5", "ours_base@euler001x5"),
    ("rk4005", "ours_base@rk4005"),
    ("rk4001x5", "ours_base@rk4001x5"),
]
RECIPE = [
    "--epochs", "60", "--lr", "0.001", "--batch_size", "256",
    "--w_roll", "0.0", "--windows_per_epoch", "0", "--ss_max", "0.5",
    "--ss_warmup", "20", "--wd", "0.00001", "--w_state", "1.0",
    "--seq_len", "8", "--future_steps", "5",
]


def job(seed, spec, key):
    name = f"{key}_s{seed}"
    return dict(
        seed=seed, spec=spec, key=key, name=name,
        ckpt_dir=os.path.join(EXTRA, name),
        log=os.path.join(EXTRA, f"train_{name}.log"),
    )


def is_complete(j):
    if not os.path.exists(os.path.join(j["ckpt_dir"], "best.pt")):
        return False
    if not os.path.exists(j["log"]):
        return False
    with open(j["log"], encoding="utf-8", errors="replace") as f:
        return "Epoch 60/60" in f.read()


def launch(j):
    cmd = [sys.executable, "-X", "utf8", "-u", "train_e0.py",
           "--arm", "ours_base", "--seed", str(j["seed"]),
           "--prior_spec", j["spec"],
           "--checkpoint_dir", os.path.relpath(j["ckpt_dir"], ROOT)] + RECIPE
    out = open(j["log"], "w", encoding="utf-8")
    err = open(j["log"] + ".err", "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err)
    j["proc"], j["out"], j["err"] = p, out, err
    return j


def finish(j):
    j["out"].close()
    j["err"].close()
    j["returncode"] = j["proc"].returncode
    j["complete"] = is_complete(j)
    tail = ""
    if os.path.exists(j["log"]):
        with open(j["log"], encoding="utf-8", errors="replace") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        tail = lines[-1] if lines else ""
    j["last_log_line"] = tail
    err_size = os.path.getsize(j["log"] + ".err") if os.path.exists(j["log"] + ".err") else 0
    j["stderr_bytes"] = err_size
    m = re.search(r"val one-step = ([\d.]+)px", tail)
    j["val_one_step_px"] = float(m.group(1)) if m else None
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[50, 51])
    ap.add_argument("--max-concurrent", type=int, default=4)
    args = ap.parse_args()

    jobs = [job(s, spec, key) for s in args.seeds for spec, key in TIERS]
    pending = []
    for j in jobs:
        if is_complete(j):
            print(f"[skip] complete: {j['name']}")
        else:
            pending.append(j)
    print(f"[plan] {len(jobs)} jobs, {len(jobs)-len(pending)} already complete, "
          f"{len(pending)} to run, max_concurrent={args.max_concurrent}")

    done, running = [], []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            j = launch(pending.pop(0))
            running.append(j)
            print(f"[start] {j['name']} (pid {j['proc'].pid})", flush=True)
        time.sleep(20)
        still = []
        for j in running:
            if j["proc"].poll() is None:
                still.append(j)
            else:
                finish(j)
                status = "OK" if j["complete"] else f"FAILED(rc={j['returncode']})"
                print(f"[done ] {j['name']} {status} val={j['val_one_step_px']} "
                      f"stderr={j['stderr_bytes']}B | {j['last_log_line']}", flush=True)
                done.append(j)
        running = still
        if running:
            el = time.time() - t0
            print(f"[wait ] running={len(running)} done={len(done)}/{len(jobs)} "
                  f"elapsed={el/60:.1f}min", flush=True)

    summary = dict(
        seeds=list(args.seeds), max_concurrent=args.max_concurrent,
        n_jobs=len(jobs), n_complete=sum(1 for j in done if j["complete"]),
        elapsed_min=round((time.time() - t0) / 60, 1),
        jobs=[dict(name=j["name"], seed=j["seed"], spec=j["spec"],
                   complete=j["complete"], returncode=j.get("returncode"),
                   val_one_step_px=j.get("val_one_step_px"),
                   stderr_bytes=j.get("stderr_bytes"),
                   last_log_line=j.get("last_log_line")) for j in done],
    )
    os.makedirs(os.path.join(ROOT, "analysis_output"), exist_ok=True)
    with open(os.path.join(ROOT, "analysis_output", "seed_extension_status.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n[summary] {summary['n_complete']}/{summary['n_jobs']} complete in "
          f"{summary['elapsed_min']} min")
    for j in summary["jobs"]:
        print(f"  {'OK  ' if j['complete'] else 'FAIL'} {j['name']:<28s} {j['val_one_step_px']}")
    print("[save] analysis_output/seed_extension_status.json")
    if summary["n_complete"] != summary["n_jobs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
