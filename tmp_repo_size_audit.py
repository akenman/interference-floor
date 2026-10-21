# -*- coding: utf-8 -*-
"""仓库体积审计（临时脚本，用完即删）：区分 master 历史 / IDE-agent 快照 refs / 工作区。"""
import subprocess
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def objects(rev_args):
    """返回 {sha: path} 与 {sha: size}。"""
    out = git("rev-list", "--objects", *rev_args)
    pairs = []
    for line in out.splitlines():
        sha, _, path = line.partition(" ")
        pairs.append((sha, path))
    if not pairs:
        return {}, {}
    chk = subprocess.run(["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize)"],
                         input="\n".join(s for s, _ in pairs), capture_output=True,
                         text=True, encoding="utf-8", errors="replace").stdout
    sizes = {}
    for line in chk.splitlines():
        p = line.split()
        if len(p) == 3 and p[0] == "blob":
            sizes[p[1]] = int(p[2])
    return dict(pairs), sizes


def mb(x):
    return x / 1048576


def summarize(name, paths, sizes, top=12):
    blob_sizes = {s: sizes.get(s, 0) for s in paths}
    total = sum(blob_sizes.values())
    per_dir = defaultdict(int)
    for s, p in paths.items():
        head = p.split("/")[0] if "/" in p else p
        per_dir[head] += blob_sizes.get(s, 0)
    print(f"\n=== {name}: {len(paths)} 对象 / blob {mb(total):,.1f} MB ===")
    for k, v in sorted(per_dir.items(), key=lambda kv: -kv[1])[:top]:
        if v > 0:
            print(f"   {mb(v):9,.1f} MB  {k}")


print("master HEAD:", git("rev-parse", "--short", "HEAD").strip())
refs = [r for r in git("for-each-ref", "--format=%(refname)").splitlines() if r]
ns = defaultdict(int)
for r in refs:
    ns["/".join(r.split("/")[:2])] += 1
print("\n=== refs 命名空间 ===")
for k, v in sorted(ns.items(), key=lambda kv: -kv[1]):
    print(f"   {v:5d}  {k}")

A, szA = objects(["master"])
ALL, szALL = objects(["--all"])
C, szC = objects(["--all", "--not", "master"])
D, szD = objects(["--all", "--not", "master", "--not", "refs/cline",
                  "--not", "refs/codex"])

summarize("master 可达（项目自身历史）", A, szA)
summarize("仅非 master ref 可达（IDE agent 快照）", C, szC)
summarize("非 master、且非 cline/codex（其他 ref）", D, szD)

print("\n=== master 历史里 >50MB 的 blob（gc 无法回收，需改写历史）===")
big = [(szA.get(s, 0), p) for s, p in A.items() if szA.get(s, 0) > 50 * 1048576]
for s, p in sorted(big, reverse=True):
    print(f"   {mb(s):9,.1f} MB  {p}")
print(f"   小计 {mb(sum(s for s, _ in big)):,.1f} MB")

print("\n=== 仅 IDE 快照持有的 >20MB 目录（删 refs 后可回收）===")
per_dir = defaultdict(int)
for s, p in C.items():
    top = p.split("/")[0] if "/" in p else p
    per_dir[top] += szC.get(s, 0)
for k, v in sorted(per_dir.items(), key=lambda kv: -kv[1]):
    if v > 20 * 1048576:
        print(f"   {mb(v):9,.1f} MB  {k}")

garbage = sum(1 for _ in git("fsck", "--no-progress").splitlines() if "tmp_obj" in _)
print(f"\n可见 tmp_obj 垃圾对象行数: {garbage}")
