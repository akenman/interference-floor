"""生成预注册时间证据清单：每个预注册文件的 SHA-256、首次入库提交，
以及对应报告工件的首次入库提交（优先使用报告 JSON 内嵌的 `prereg` 指向，
无内嵌指向时按槽位前缀回退），用于证明预注册先于运行报告的入库。

输出 analysis_output/prereg_manifest.json（随投稿包发布）。
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "analysis_output", "prereg_manifest.json")


def git(*args):
    r = subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr))
    return r.stdout.strip()


def intro_commit(path):
    out = git("log", "--diff-filter=A", "--follow", "--format=%h|%aI", "--", path)
    lines = [l for l in out.splitlines() if "|" in l]
    if not lines:
        return None, None
    sha, date = lines[-1].split("|", 1)
    return sha, date


def norm(path):
    return path.replace("\\", "/").lstrip("./")


def load_refs():
    """报告 JSON 内嵌的 prereg 指向：prereg path -> [report paths]。"""
    refs = {}
    for f in git("ls-files", "analysis_output").splitlines():
        if not f.endswith(".json") or "prereg" in os.path.basename(f).lower():
            continue
        try:
            with io.open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        p = d.get("prereg")
        if isinstance(p, str):
            refs.setdefault(norm(p), []).append(norm(f))
    return refs


def slot_prefix(path):
    m = re.search(r"prereg_([A-Za-z0-9]+?)(?:_[a-z]|\.json$)", os.path.basename(path))
    return m.group(1) if m else None


MANUAL_ARTIFACTS = {
    "experiments/prereg_P0a_prior_ladder.json": ["analysis_output/E0_report.json"],
}


def main():
    refs = load_refs()
    tracked = git("ls-files", "experiments/prereg_*.json").splitlines()
    doc = "paper/PREREG_CRITERIA_EN.md"
    all_files = git("ls-files").splitlines()
    entries = []
    for path in tracked + [doc]:
        path = norm(path)
        with open(os.path.join(ROOT, path), "rb") as fh:
            blob = fh.read()
        sha, date = intro_commit(path)
        with io.open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            meta = json.load(fh) if path.endswith(".json") else {}
        entry = {
            "file": path,
            "role": "slot" if path.endswith(".json") else "criteria-summary",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "declared_written_at": meta.get("prereg_written_at"),
            "intro_commit": sha,
            "intro_date": date,
        }
        arts = list(refs.get(path, [])) or list(MANUAL_ARTIFACTS.get(path, []))
        source = "embedded-prereg-ref"
        if not arts:
            pref = slot_prefix(path)
            if pref:
                arts = [norm(f) for f in all_files
                        if os.path.basename(f).lower().startswith(pref.lower())
                        and f.endswith(".json")
                        and "prereg" not in os.path.basename(f).lower()]
                source = "filename-prefix"
        ordered = []
        for f in arts:
            rsha, rdate = intro_commit(f)
            if rdate:
                ordered.append({"file": f, "intro_commit": rsha,
                                "intro_date": rdate})
        ordered.sort(key=lambda r: r["intro_date"])
        entry["report_artifacts"] = ordered[:10]
        entry["report_artifacts_total"] = len(ordered)
        entry["artifact_matching"] = source if ordered else None
        if ordered and date:
            pre_day = date[:10]
            rep_day = ordered[0]["intro_date"][:10]
            if pre_day < rep_day:
                ordering = "prereg-earlier"
            elif pre_day == rep_day:
                ordering = "same-day"
            else:
                ordering = "report-earlier"
            entry["ordering"] = ordering
            entry["prereg_before_first_report"] = ordering != "report-earlier"
        else:
            entry["ordering"] = None
            entry["prereg_before_first_report"] = None
        entry["pilot_calibration_declared"] = "pilot_calibration" in meta
        entries.append(entry)

    slots = [e for e in entries if e["role"] == "slot"]
    ok = [e for e in slots if e["prereg_before_first_report"] is True]
    bad = [e for e in slots if e["prereg_before_first_report"] is False]
    unknown = [e for e in slots if e["prereg_before_first_report"] is None]
    out = {
        "note": "SHA-256 and introduction commits of the preregistration files. "
                "Report artifacts are mapped by the report JSON's embedded "
                "`prereg` field, else by slot filename prefix; dates are git "
                "author dates of each file's first commit.",
        "generated_by": "tmp_prereg_manifest.py",
        "n_files": len(entries),
        "slots_before_first_report": len(ok),
        "ordering_granularity": "day (git author date)",
        "slots_after_first_report": len(bad),
        "slots_unknown": len(unknown),
        "entries": entries,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("wrote", OUT)
    print("%d files; %d slots before; %d after; %d unknown"
          % (len(entries), len(ok), len(bad), len(unknown)))
    for e in ok:
        if e["ordering"] == "same-day":
            print("  same-day:", e["file"], "| pilot declared:",
                  e["pilot_calibration_declared"])
    for e in bad + unknown:
        print("  CHECK:", e["file"], e["prereg_before_first_report"],
              "| artifacts:", e["report_artifacts_total"],
              "| match:", e["artifact_matching"])
        for r in e["report_artifacts"][:2]:
            print("     report:", r["file"], r["intro_date"])


if __name__ == "__main__":
    sys.exit(main())
