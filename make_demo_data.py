# -*- coding: utf-8 -*-
"""
生成与竞赛附件同格式的 365 天合成数据，用于在没有原始附件时也能跑通整个程序。

输出（写到脚本上一级目录，即仓库根目录）：
    附件1(2).xlsx   单日基准：时间 / 电价 / 小区负载 / 光伏预测功率，144 行 4 列
    附件2(2).xlsx   两个 sheet：小区负载、光伏发电实际功率，宽表 date + 144 列
    附件3(2).xlsx   date / release / 未来 24 小时光伏预测，1460 行
    附件4(2).xlsx   宽表 date + 144 列动态电价

数据全部为合成数据，仅用于验证代码逻辑，不代表任何真实电网数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
SLOTS = 144
DAYS = 365
START = pd.Timestamp("2025-01-01")
TIME_LABELS = [f"{i // 6}:{(i % 6) * 10:02d}" for i in range(SLOTS)]


def tou_price() -> np.ndarray:
    """峰谷平分时电价：谷 0.30 / 平 0.62 / 峰 1.05 元每 kWh。"""
    p = np.full(SLOTS, 0.62)
    p[0:42] = 0.30        # 00:00-07:00 谷
    p[42:54] = 0.62       # 07:00-09:00 平
    p[54:72] = 1.05       # 09:00-12:00 峰
    p[72:84] = 0.62       # 12:00-14:00 平
    p[84:102] = 1.05      # 14:00-17:00 峰
    p[102:120] = 1.05     # 17:00-20:00 峰
    p[120:138] = 0.62     # 20:00-23:00 平
    p[138:144] = 0.30     # 23:00-24:00 谷
    return p


def load_profile(day_index: int) -> np.ndarray:
    """双峰型居民负荷，夏季整体偏高，叠加随机扰动。"""
    h = np.arange(SLOTS) / 6.0
    seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (day_index - 105) / 365.0)
    base = 1900.0 * seasonal
    morning = 900.0 * np.exp(-((h - 8.0) ** 2) / (2 * 1.5 ** 2))
    evening = 1700.0 * np.exp(-((h - 19.5) ** 2) / (2 * 2.0 ** 2))
    noise = RNG.normal(0.0, 120.0, SLOTS)
    return np.maximum(base + morning + evening + noise, 300.0)


def pv_profile(day_index: int, cloud: float = 1.0) -> np.ndarray:
    """单峰光伏出力，6:00 起、12:30 附近见顶、19:00 归零。"""
    h = np.arange(SLOTS) / 6.0
    seasonal = 1.0 + 0.30 * np.sin(2 * np.pi * (day_index - 80) / 365.0)
    shape = np.where((h >= 6.0) & (h <= 19.0), np.sin(np.pi * (h - 6.0) / 13.0), 0.0)
    peak = 4200.0 * seasonal
    noise = RNG.normal(0.0, 60.0, SLOTS)
    return np.maximum(peak * shape * cloud + noise, 0.0)


def hourly(power: np.ndarray) -> np.ndarray:
    """10 分钟序列压缩为 24 个整点均值。"""
    return power.reshape(24, 6).mean(axis=1)


def main() -> None:
    from pathlib import Path

    out_dir = Path(__file__).resolve().parent.parent
    price = tou_price()

    # 附件1：基准日
    d1_load = load_profile(0)
    d1_pv = pv_profile(0, cloud=0.85)
    df1 = pd.DataFrame({
        "时间": TIME_LABELS,
        "电价(元/kWh)": price,
        "小区负载(kW)": np.round(d1_load, 2),
        "光伏发电预测功率(kW)": np.round(d1_pv, 2),
    })
    df1.to_excel(out_dir / "附件1(2).xlsx", index=False)

    # 附件2：全年实际负荷与光伏
    load_rows, pv_rows, price_rows = [], [], []
    dates = []
    for i in range(DAYS):
        day = START + pd.Timedelta(days=i)
        dates.append(day)
        cloud = float(np.clip(RNG.normal(0.85, 0.22), 0.25, 1.15))
        load_rows.append(np.round(load_profile(i), 2).tolist())
        pv_rows.append(np.round(pv_profile(i, cloud=cloud), 2).tolist())
        # 附件4：动态电价 = 分时电价 × 当日系数
        coef = float(np.clip(RNG.normal(1.0, 0.12), 0.75, 1.30))
        price_rows.append(np.round(price * coef, 4).tolist())

    cols = TIME_LABELS
    df2_load = pd.DataFrame(load_rows, columns=cols)
    df2_load.insert(0, "日期", dates)
    df2_pv = pd.DataFrame(pv_rows, columns=cols)
    df2_pv.insert(0, "日期", dates)
    with pd.ExcelWriter(out_dir / "附件2(2).xlsx", engine="openpyxl") as w:
        df2_load.to_excel(w, sheet_name="小区负载", index=False)
        df2_pv.to_excel(w, sheet_name="光伏发电实际功率", index=False)

    df4 = pd.DataFrame(price_rows, columns=cols)
    df4.insert(0, "日期", dates)
    df4.to_excel(out_dir / "附件4(2).xlsx", index=False)

    # 附件3：每天 4 个发布时刻 × 未来 24 小时预测
    recs = []
    for i in range(DAYS):
        day = START + pd.Timedelta(days=i)
        actual_h = hourly(pv_profile(i))
        for r in (0, 6, 12, 18):
            if r == 12:
                # 12:00 之后的预报更接近真实（午后云量已可观测）
                bias, sigma = 0.02, 0.10
            elif r == 18:
                bias, sigma = 0.05, 0.14
            elif r == 6:
                bias, sigma = 0.08, 0.20
            else:
                bias, sigma = 0.10, 0.28
            vals = []
            for k in range(1, 25):
                tgt = (r + k) % 24
                noisy = actual_h[tgt] * (1 + bias) + RNG.normal(0.0, sigma * actual_h[tgt] + 30.0)
                vals.append(round(max(float(noisy), 0.0), 2))
            recs.append({"日期": day, "发布时刻": f"{r}:00", **{f"{h + 1}": v for h, v in enumerate(vals)}})

    df3 = pd.DataFrame(recs)
    df3.to_excel(out_dir / "附件3(2).xlsx", index=False)

    print("已生成合成数据：")
    for name in ["附件1(2).xlsx", "附件2(2).xlsx", "附件3(2).xlsx", "附件4(2).xlsx"]:
        print("  ", out_dir / name)


if __name__ == "__main__":
    main()
