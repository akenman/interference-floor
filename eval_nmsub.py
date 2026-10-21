# -*- coding: utf-8 -*-
"""细化先验积分器模型（T19 判决性附录：5×0.01s 半隐式欧拉，同方程同参数）。"""
import torch
from benchmark_surpass import OursParamModel


class OursSubModel(OursParamModel):
    """先验积分器细化版：0.05s = 5×0.01s 半隐式欧拉（消除 O(dt) 与 g 的混叠）。"""

    def prior_step(self, obj_norm, a_norm):
        g = self.log_g.exp().clamp(500, 1500)
        m1 = self.log_M1.exp().clamp(0.1, 10)
        m2 = self.log_M2.exp().clamp(0.1, 10)
        l1 = self.log_L1.exp().clamp(20, 200)
        l2 = self.log_L2.exp().clamp(20, 200)
        raw = obj_norm * (self.std_obj + 1e-8) + self.mean_obj
        b1, b2 = 0, 11
        th1 = torch.atan2(raw[:, b1 + 3], raw[:, b1 + 4])
        th2 = torch.atan2(raw[:, b2 + 3], raw[:, b2 + 4])
        w1 = raw[:, b1 + 7]
        w2 = raw[:, b2 + 7]
        tau = a_norm.squeeze(-1) * 28000.0
        dt = 0.01
        for _ in range(5):
            c2, s2 = torch.cos(th2), torch.sin(th2)
            M11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2
            M12 = m2 * l2 ** 2 + m2 * l1 * l2 * c2
            M22 = m2 * l2 ** 2
            h1 = (-m2 * l1 * l2 * (2 * w1 + w2) * w2 * s2
                  + (m1 + m2) * g * l1 * torch.sin(th1)
                  + m2 * g * l2 * torch.sin(th1 + th2))
            h2 = (m2 * l1 * l2 * w1 ** 2 * s2 + m2 * g * l2 * torch.sin(th1 + th2))
            det = M11 * M22 - M12 ** 2
            a1 = (M22 * (-h1) - M12 * (-h2)) / det
            a2 = (-M12 * (-h1) + M11 * (-h2)) / det
            w1 = w1 + a1 * dt
            w2 = w2 + a2 * dt
            th1 = th1 + w1 * dt
            th2 = th2 + w2 * dt
        out = raw.clone()
        x1 = l1 * torch.sin(th1); y1 = -l1 * torch.cos(th1)
        x2 = x1 + l2 * torch.sin(th1 + th2)
        y2 = y1 - l2 * torch.cos(th1 + th2)
        vx1 = l1 * w1 * torch.cos(th1); vy1 = l1 * w1 * torch.sin(th1)
        vx2 = vx1 + l2 * (w1 + w2) * torch.cos(th1 + th2)
        vy2 = vy1 + l2 * (w1 + w2) * torch.sin(th1 + th2)
        out[:, b1 + 1], out[:, b1 + 2] = x1, y1
        out[:, b2 + 1], out[:, b2 + 2] = x2, y2
        out[:, b1 + 3], out[:, b1 + 4] = torch.sin(th1), torch.cos(th1)
        out[:, b2 + 3], out[:, b2 + 4] = torch.sin(th2), torch.cos(th2)
        out[:, b1 + 5], out[:, b1 + 6] = vx1, vy1
        out[:, b2 + 5], out[:, b2 + 6] = vx2, vy2
        out[:, b1 + 7], out[:, b2 + 7] = w1, w2
        return (out - self.mean_obj) / (self.std_obj + 1e-8)
