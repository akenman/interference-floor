"""E0 主表修复后的接线 + 前向 smoke test（覆盖全部 9 臂）。"""
import torch
import sys
sys.path.insert(0, ".")
from model_e0 import make_model, ARM_SPECS, assert_wiring

B, L, K = 2, 8, 5
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'臂':12s} {'h':>12} {'d':>12} rollout")
all_ok = True
for arm in ARM_SPECS:
    m = make_model(arm).to(device)
    x = torch.randn(B, L, 25, device=device)
    a = torch.randn(B, K, 1, device=device)
    h, c = m.encode_sequence(x)
    fn = m.predict_state if hasattr(m, "predict_state") else m.predict
    d = fn(h, a[:, 0])
    out, _ = m.rollout(x, a, x_fut_gt=torch.randn(B, K, 22, device=device), ss_prob=0.0)
    tag = "OK" if (list(d.shape) == [B, 22] and all(list(o.shape) == [B, 22] for o in out)) else "FAIL"
    if tag != "OK":
        all_ok = False
    print(f"{arm:12s} {str(list(h.shape)):>12} {str(list(d.shape)):>12} {tag}")

print()
print("接线断言:")
for arm in ARM_SPECS:
    ok, probs = assert_wiring(arm, make_model(arm).state_dict())
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {arm}" + (f"  {probs}" if probs else ""))

print()
print("ALL PASS" if all_ok else "FAILURES PRESENT")