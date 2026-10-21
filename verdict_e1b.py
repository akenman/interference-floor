# -*- coding: utf-8 -*-
"""
E1b 判决脚本 (verdict_e1b.py)
==================================================
输入: analysis_output/E1b_raw.json
输出: 三条预注册判据的判决（JSON + 打印）

判决逻辑完全机械化，不依赖人工解读。最难/最易卦由 B-uniform 的经验排序决定。

用法: python verdict_e1b.py
"""

import json
import os

GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    raw = load("analysis_output/E1b_raw.json")
    arms = raw["arms"]

    # ---- 最难/最易卦：由 B-uniform 经验排序决定 ---------------------
    ge = arms["uniform"]["gua_err_mean"]
    order = sorted(range(8), key=lambda g: -ge[f"g{g} ({GUA_NAMES[g]})"])
    hardest, easiest = order[0], order[-1]
    hardest_name = GUA_NAMES[hardest]
    easiest_name = GUA_NAMES[easiest]
    print(f"经验难度排序（B-uniform, 难→易）: "
          + " > ".join(f"{GUA_NAMES[g]}({ge[f'g{g} ({GUA_NAMES[g]})']:.3f})" for g in order))
    print(f"→ 最难 = {hardest_name} (g{hardest}), 最易 = {easiest_name} (g{easiest})")

    u = arms["uniform"]
    p = arms["priority"]
    ph = arms["phase"]

    def gerr(arm, g_idx):
        return arm["gua_err_mean"][f"g{g_idx} ({GUA_NAMES[g_idx]})"]

    # ---------- E1b-1: B-priority 整体 < B-uniform，差距 ≥ 3% ----------
    uni_overall = u["overall_mean"]
    pri_overall = p["overall_mean"]
    rel_drop = (uni_overall - pri_overall) / uni_overall
    e1b1_pass = rel_drop >= 0.03
    print(f"\n[E1b-1] uniform={uni_overall:.4f}  priority={pri_overall:.4f}  "
          f"相对下降={rel_drop*100:+.2f}%  → {'[OK] 通过' if e1b1_pass else '[FAIL] 未过'} (门槛 ≥3%)")

    # ---------- E1b-2: 难卦降幅 > 易卦升幅（净收益）-------------------
    u_hard, p_hard = gerr(u, hardest), gerr(p, hardest)
    u_easy, p_easy = gerr(u, easiest), gerr(p, easiest)
    delta_hard = p_hard - u_hard          # 负 = 下降
    delta_easy = p_easy - u_easy          # 正 = 上升
    net = delta_hard + delta_easy         # 总卦级误差变化
    e1b2_pass = net < 0
    print(f"[E1b-2] {hardest_name}: {u_hard:.4f} → {p_hard:.4f} (Δ={delta_hard:+.4f})  "
          f"| {easiest_name}: {u_easy:.4f} → {p_easy:.4f} (Δ={delta_easy:+.4f})  "
          f"| 净变化={net:+.4f} → {'[OK] 通过' if e1b2_pass else '[FAIL] 未过'} (门槛: 净变化<0)")

    # ---------- E1b-3: B-phase 难卦 < B-uniform 难卦 ------------------
    ph_hard = gerr(ph, hardest)
    e1b3_pass = ph_hard < u_hard
    print(f"[E1b-3] {hardest_name}  uniform={u_hard:.4f}  phase={ph_hard:.4f}  "
          f"→ {'[OK] 通过' if e1b3_pass else '[FAIL] 未过'} (门槛: 严格更小)")

    # ---- 预注册身份的交叉核对 ---------------------------------------
    pre_hardest, pre_easiest = 5, 0          # 预注册: 离 / 坤
    identity_match = (hardest == pre_hardest and easiest == pre_easiest)
    print(f"\n[身份核对] 预注册 难={GUA_NAMES[pre_hardest]}/易={GUA_NAMES[pre_easiest]}  "
          f"vs 经验 难={hardest_name}/易={easiest_name} → "
          f"{'一致' if identity_match else '不一致（以经验排序为准重算 E1b-2）'}")

    verdict = {
        "hardest_gua": {"index": hardest, "name": hardest_name, "error_uniform": round(u_hard, 4),
                        "error_priority": round(p_hard, 4), "error_phase": round(ph_hard, 4)},
        "easiest_gua": {"index": easiest, "name": easiest_name, "error_uniform": round(u_easy, 4),
                        "error_priority": round(p_easy, 4)},
        "overall": {"uniform": round(uni_overall, 4), "priority": round(pri_overall, 4),
                    "phase": round(ph["overall_mean"], 4)},
        "E1b-1": {"pass": bool(e1b1_pass), "actual": f"{rel_drop * 100:+.2f}%",
                  "threshold": "≥3%",
                  "note": f"B-priority {pri_overall:.4f} vs B-uniform {uni_overall:.4f}"},
        "E1b-2": {"pass": bool(e1b2_pass), "actual": f"净变化={net:+.4f}px",
                  "threshold": "净卦级误差变化 < 0",
                  "note": f"{hardest_name} Δ={delta_hard:+.4f}, {easiest_name} Δ={delta_easy:+.4f}"},
        "E1b-3": {"pass": bool(e1b3_pass), "actual": f"{ph_hard:.4f} < {u_hard:.4f}"
                  if e1b3_pass else f"{ph_hard:.4f} ≮ {u_hard:.4f}",
                  "threshold": "B-phase 难卦误差严格更小",
                  "note": f"{hardest_name}: phase {ph_hard:.4f}, uniform {u_hard:.4f}"},
        "gua_identity_crosscheck": {
            "pre_registered": f"难={GUA_NAMES[pre_hardest]}/易={GUA_NAMES[pre_easiest]}",
            "empirical": f"难={hardest_name}/易={easiest_name}",
            "matches": identity_match,
        },
    }

    os.makedirs("analysis_output", exist_ok=True)
    with open("analysis_output/E1b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 判决已写入 analysis_output/E1b_verdict.json")


if __name__ == "__main__":
    main()