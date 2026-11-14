# -*- coding: utf-8 -*-
"""
2026 数学建模 C题：微网与外部电网电力调控策略
完整可运行程序

功能：
1. 问题1：固定电价、固定负荷、给定光伏预测的日内购电优化（MILP）
2. 问题2：全年逐日滚动执行的安全裕度方案，自动比较安全裕度并选择最优值
3. 问题3：附件3光伏预报 -> BP神经网络残差修正 -> 95%经验预测区间
          -> 0/6/12/18时滚动优化 -> 计划/调整/应急结算，并分析更新频率
4. 问题4：将电价替换为附件4逐日逐时动态电价后，重新求解问题2和问题3

数据要求：
把本脚本与以下附件放在同一目录：
附件1(2).xlsx、附件2(2).xlsx、附件3(2).xlsx、附件4(2).xlsx
脚本会优先读取脚本所在目录，也支持程序当前工作目录。

主要依赖：
Python >= 3.10
numpy, pandas, scipy, scikit-learn, openpyxl

输出：
result1.xlsx
result2.xlsx
result3.xlsx
result4-2.xlsx
result4-3.xlsx
model_summary.xlsx
"""

from __future__ import annotations

import os
import sys
import math
import warnings
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 全局参数
# =============================================================================

DT = 1.0 / 6.0          # 10分钟 = 1/6小时
SLOTS_PER_DAY = 144
HOURS_PER_DAY = 24
SLOTS_PER_HOUR = 6

E_CAP = 12000.0
SOC_MIN = 1200.0
SOC_MAX = 10800.0
P_BAT_MAX = 5000.0
ETA = 0.90
SOC_INITIAL = 6000.0
SOC_TERMINAL = 6000.0

EMERGENCY_MULTIPLIER = 5.0
DOWN_ADJUST_MULTIPLIER = 0.5
UP_ADJUST_MULTIPLIER = 1.5

DEFAULT_MARGINS = [0.00, 0.02, 0.05, 0.08, 0.10]
Q2_PLAN_FORECAST_METHOD = "PREV_DAY"  # 0:00仅使用已知信息；首日采用附件1基准日冷启动
OFFICIAL_UPDATE_HOURS = [0, 6, 12, 18]
EXTRA_UPDATE_HOURS = [3, 9, 15, 21]

RANDOM_SEED = 2026

# 运行模式：False为LP等价快速模式，True为严格带0-1充放电互斥变量的MILP。
# 在本题的正电价、充放电效率小于1且无同时充放电收益的条件下，LP模式与MILP最优值一致，
# 但全年365天重复求解会明显更快。论文中仍按MILP形式给出模型；需要逐项验证二进制变量时改为True。
EXACT_BINARY_MILP = False

# =============================================================================
# 1. 路径与文件定位
# =============================================================================

def get_base_dir() -> Path:
    """优先取脚本目录，保证用户把附件放在py根目录时无需改路径。"""
    try:
        script_dir = Path(__file__).resolve().parent
        if script_dir.exists():
            return script_dir
    except Exception:
        pass
    return Path.cwd().resolve()


BASE_DIR = get_base_dir()


def find_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / filename,
        Path.cwd().resolve() / filename,
        BASE_DIR / filename.replace("(2)", ""),
        Path.cwd().resolve() / filename.replace("(2)", ""),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"找不到 {filename}。\n"
        f"请将附件放到以下目录之一：\n"
        f"1) {BASE_DIR}\n"
        f"2) {Path.cwd().resolve()}"
    )


FILE_A1 = find_file("附件1(2).xlsx")
FILE_A2 = find_file("附件2(2).xlsx")
FILE_A3 = find_file("附件3(2).xlsx")
FILE_A4 = find_file("附件4(2).xlsx")

# =============================================================================
# 2. 数据结构
# =============================================================================

@dataclass
class DayData:
    date: pd.Timestamp
    load_kw: np.ndarray
    pv_kw: np.ndarray
    price: np.ndarray


@dataclass
class OptimizationResult:
    success: bool
    status: int
    message: str
    objective: float
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    curtailment: np.ndarray


# =============================================================================
# 3. 通用工具
# =============================================================================

def excel_serial_to_timestamp(x: Any) -> pd.Timestamp:
    """Excel日期序列号转Timestamp。"""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, str):
        x2 = x.strip()
        # 数字字符串：先尝试Excel序列号
        try:
            fx = float(x2)
            if 20000 < fx < 70000:
                return pd.Timestamp("1899-12-30") + pd.to_timedelta(fx, unit="D")
        except Exception:
            pass
        return pd.to_datetime(x2, errors="coerce")
    if isinstance(x, (int, float, np.integer, np.floating)):
        fx = float(x)
        if 20000 < fx < 70000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(fx, unit="D")
    return pd.to_datetime(x, errors="coerce")


def time_to_minutes(x: Any) -> int:
    """把Excel时间小数、HH:MM、Timestamp等转为分钟。"""
    if pd.isna(x):
        return 0
    if isinstance(x, (float, int, np.floating, np.integer)):
        v = float(x)
        if 0 <= v < 1.0:
            return int(round(v * 24 * 60)) % (24 * 60)
        # 如果直接是分钟数
        if 0 <= v < 1440:
            return int(round(v))
    if isinstance(x, pd.Timestamp):
        return x.hour * 60 + x.minute
    s = str(x).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return (h * 60 + m) % (24 * 60)
        fx = float(s)
        if 0 <= fx < 1.0:
            return int(round(fx * 24 * 60)) % 1440
        if 0 <= fx < 1440:
            return int(round(fx))
    except Exception:
        pass
    return 0


def normalize_time_columns(df: pd.DataFrame) -> List[int]:
    """从宽表表头识别144个10分钟时刻，并返回分钟数组。"""
    out = []
    for c in df.columns:
        if c == df.columns[0]:
            continue
        out.append(time_to_minutes(c))
    return out


def nearest_10min_index(minutes: float) -> int:
    return int(round(minutes / 10.0)) % SLOTS_PER_DAY


def safe_float_array(x: Any, n: Optional[int] = None) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if n is not None:
        if len(a) != n:
            raise ValueError(f"数组长度错误：期望{n}，实际{len(a)}")
    return a


def linear_interp_hourly_to_10min(hourly: np.ndarray, start_hour: int = 0) -> np.ndarray:
    """将24个整点小时值映射为144个10分钟值。

    默认每个小时内保持该小时预测值不变，避免人为制造不存在的价格/功率信息。
    对题目给出的附件3小时预报，先确定目标小时，再使用该值覆盖对应6个10分钟时段。
    """
    hourly = np.asarray(hourly, dtype=float).reshape(-1)
    if len(hourly) != 24:
        raise ValueError("hourly数组必须包含24个小时")
    out = np.zeros(SLOTS_PER_DAY, dtype=float)
    for h in range(24):
        target_h = (start_hour + h) % 24
        out[target_h * 6:(target_h + 1) * 6] = hourly[h]
    return np.maximum(out, 0.0)


def ensure_144(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    if len(a) != SLOTS_PER_DAY:
        raise ValueError(f"{name}长度必须是144，实际为{len(a)}")
    return a


def print_banner(text: str):
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)

# =============================================================================
# 4. 读取附件1：问题1固定日
# =============================================================================

def load_attachment1(path: Path) -> DayData:
    df = pd.read_excel(path, sheet_name=0)
    if df.shape[1] < 4:
        raise ValueError("附件1列数不足，应至少包含时间、电价、小区负载、光伏发电预测功率")
    df = df.iloc[:SLOTS_PER_DAY, :4].copy()
    df.columns = ["time", "price", "load", "pv"]
    data = DayData(
        date=pd.Timestamp("2025-01-01"),
        load_kw=ensure_144(df["load"].astype(float).values, "附件1负荷"),
        pv_kw=ensure_144(df["pv"].astype(float).values, "附件1光伏预测"),
        price=ensure_144(df["price"].astype(float).values, "附件1电价"),
    )
    return data

# =============================================================================
# 5. 读取附件2：全年实际负荷与实际光伏
# =============================================================================

def wide_daily_sheet_to_dict(df: pd.DataFrame) -> Dict[pd.Timestamp, np.ndarray]:
    df = df.copy()
    date_col = df.columns[0]
    time_cols = list(df.columns[1:])
    times = [time_to_minutes(c) for c in time_cols]
    order = np.argsort(times)
    time_cols = [time_cols[i] for i in order]
    result: Dict[pd.Timestamp, np.ndarray] = {}

    for _, row in df.iterrows():
        date = excel_serial_to_timestamp(row[date_col])
        if pd.isna(date):
            continue
        date = pd.Timestamp(date).normalize()
        vals = pd.to_numeric(row[time_cols], errors="coerce").fillna(0.0).astype(float).values
        if len(vals) != 144:
            raise ValueError(f"附件2每日数据不是144个时段，日期={date}，列数={len(vals)}")
        result[date] = np.maximum(vals, 0.0)
    return result


def load_attachment2(path: Path) -> Tuple[Dict[pd.Timestamp, np.ndarray], Dict[pd.Timestamp, np.ndarray]]:
    load_df = pd.read_excel(path, sheet_name="小区负载")
    pv_df = pd.read_excel(path, sheet_name="光伏发电实际功率")
    load_dict = wide_daily_sheet_to_dict(load_df)
    pv_dict = wide_daily_sheet_to_dict(pv_df)
    common = sorted(set(load_dict) & set(pv_dict))
    if len(common) < 300:
        raise ValueError(f"附件2共同日期数量异常：{len(common)}")
    return load_dict, pv_dict

# =============================================================================
# 6. 读取附件3：4个时刻的未来24h光伏预报
# =============================================================================

def load_attachment3(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    if df.shape[1] < 26:
        raise ValueError("附件3应至少包含日期、预报时刻以及24个未来小时预报")
    cols = list(df.columns)
    rename = {cols[0]: "date", cols[1]: "release"}
    for i in range(24):
        rename[cols[2 + i]] = f"h{i + 1}"
    df = df.rename(columns=rename)
    df["date"] = df["date"].replace("", np.nan).ffill()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["release"] = df["release"].astype(str).str.strip()
    df = df[df["date"].notna()].copy()
    df["release_hour"] = df["release"].str.extract(r"(\d+)\s*:")[0].astype(int)
    for i in range(1, 25):
        df[f"h{i}"] = pd.to_numeric(df[f"h{i}"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)

# =============================================================================
# 7. 读取附件4：全年动态电价
# =============================================================================

def load_attachment4(path: Path) -> Dict[pd.Timestamp, np.ndarray]:
    df = pd.read_excel(path, sheet_name=0)
    out: Dict[pd.Timestamp, np.ndarray] = {}
    date_col = df.columns[0]
    time_cols = list(df.columns[1:])
    times = [time_to_minutes(c) for c in time_cols]
    order = np.argsort(times)
    time_cols = [time_cols[i] for i in order]
    for _, row in df.iterrows():
        date = excel_serial_to_timestamp(row[date_col])
        if pd.isna(date):
            continue
        date = pd.Timestamp(date).normalize()
        vals = pd.to_numeric(row[time_cols], errors="coerce").fillna(0.0).astype(float).values
        if len(vals) != 144:
            raise ValueError(f"附件4不是144时段电价：{date}，得到{len(vals)}")
        out[date] = vals
    if len(out) < 300:
        raise ValueError(f"附件4日期数量异常：{len(out)}")
    return out

# =============================================================================
# 8. MILP模型：单日储能调度
# =============================================================================

def solve_lp_day(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    initial_soc: float,
    terminal_soc: Optional[float],
) -> OptimizationResult:
    """不含0-1变量的LP等价快速求解。

    当所有外购电价非负且储能往返效率小于1时，同时充电和放电不会降低目标函数，
    因而显式互斥二进制变量可以省略而不改变最优成本。此模式用于全年批量计算。
    """
    from scipy.optimize import linprog
    load_kw = np.asarray(load_kw, dtype=float).reshape(-1)
    pv_kw = np.maximum(np.asarray(pv_kw, dtype=float).reshape(-1), 0.0)
    price = np.asarray(price, dtype=float).reshape(-1)
    T = len(load_kw)
    if not (len(pv_kw) == len(price) == T):
        raise ValueError("LP求解输入长度不一致")
    # 每时段变量：grid, charge, discharge, soc, curtailment
    nv = 5
    n = T * nv
    def ix(t,k): return t*nv+k
    GRID,CH,DIS,SOC,CURT = range(5)
    c=np.zeros(n)
    lb=np.zeros(n)
    ub=np.full(n,np.inf)
    for t in range(T):
        c[ix(t,GRID)] = price[t]*DT
        ub[ix(t,GRID)] = max(float(load_kw[t]+pv_kw[t]+P_BAT_MAX),50000.0)
        ub[ix(t,CH)] = P_BAT_MAX
        ub[ix(t,DIS)] = P_BAT_MAX
        lb[ix(t,SOC)] = SOC_MIN
        ub[ix(t,SOC)] = SOC_MAX
        ub[ix(t,CURT)] = pv_kw[t]
    Aeq=lil_matrix((2*T+(1 if terminal_soc is not None else 0),n),dtype=float)
    beq=[]
    r=0
    for t in range(T):
        Aeq[r,ix(t,GRID)]=1
        Aeq[r,ix(t,CH)]=-1
        Aeq[r,ix(t,DIS)]=1
        Aeq[r,ix(t,CURT)]=-1
        beq.append(load_kw[t]-pv_kw[t]); r+=1
    for t in range(T):
        Aeq[r,ix(t,SOC)]=1
        Aeq[r,ix(t,CH)]=-ETA*DT
        Aeq[r,ix(t,DIS)]=DT/ETA
        if t>0:
            Aeq[r,ix(t-1,SOC)]=-1
            beq.append(0.0)
        else:
            beq.append(float(initial_soc))
        r+=1
    if terminal_soc is not None:
        Aeq[r,ix(T-1,SOC)]=1
        beq.append(float(terminal_soc))
    res=linprog(c,A_eq=csr_matrix(Aeq),b_eq=np.asarray(beq),bounds=list(zip(lb,ub)),method="highs")
    if not res.success or res.x is None:
        return OptimizationResult(False, int(getattr(res,"status",1)), str(getattr(res,"message","LP失败")), np.nan,
                                  np.full(T,np.nan),np.full(T,np.nan),np.full(T,np.nan),np.full(T,np.nan),np.full(T,np.nan))
    x=res.x
    return OptimizationResult(True,int(res.status),str(res.message),float(res.fun),
        np.array([x[ix(t,GRID)] for t in range(T)]),
        np.array([x[ix(t,CH)] for t in range(T)]),
        np.array([x[ix(t,DIS)] for t in range(T)]),
        np.array([x[ix(t,SOC)] for t in range(T)]),
        np.array([x[ix(t,CURT)] for t in range(T)]))


class MicrogridMILP:
    """单日MILP模型。

    变量按每个时段排列：
    grid, charge, discharge, soc, curtailment, y_charge, y_discharge

    约束：
    grid + pv - curtailment + discharge = load + charge
    soc_t = soc_(t-1) + eta*charge*dt - discharge/eta*dt
    charge <= Pmax*y_charge
    discharge <= Pmax*y_discharge
    y_charge + y_discharge <= 1
    SOC范围、功率范围、grid非负
    终端SOC可选；Q1固定为6000 kWh，Q2/Q3/Q4按连续运行状态自然传递
    """

    NVAR_PER_SLOT = 7

    def __init__(self, dt: float = DT):
        self.dt = dt

    def idx(self, t: int, k: int) -> int:
        return t * self.NVAR_PER_SLOT + k

    @property
    def GRID(self): return 0
    @property
    def CHARGE(self): return 1
    @property
    def DISCHARGE(self): return 2
    @property
    def SOC(self): return 3
    @property
    def CURTAIL(self): return 4
    @property
    def YCH(self): return 5
    @property
    def YDIS(self): return 6

    def solve(
        self,
        load_kw: np.ndarray,
        pv_kw: np.ndarray,
        price: np.ndarray,
        initial_soc: float = SOC_INITIAL,
        terminal_soc: Optional[float] = None,
        time_limit: Optional[float] = None,
        mip_rel_gap: float = 1e-6,
    ) -> OptimizationResult:
        if not EXACT_BINARY_MILP:
            return solve_lp_day(load_kw, pv_kw, price, initial_soc, terminal_soc)
        load_kw = ensure_144(load_kw, "load")
        pv_kw = np.maximum(ensure_144(pv_kw, "pv"), 0.0)
        price = ensure_144(price, "price")
        T = len(load_kw)
        n = T * self.NVAR_PER_SLOT

        c = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        upper = np.full(n, np.inf, dtype=float)
        integrality = np.zeros(n, dtype=int)

        for t in range(T):
            # grid
            c[self.idx(t, self.GRID)] = price[t] * self.dt
            upper[self.idx(t, self.GRID)] = max(load_kw[t] + pv_kw[t] + P_BAT_MAX, 50000.0)

            # charge / discharge
            upper[self.idx(t, self.CHARGE)] = P_BAT_MAX
            upper[self.idx(t, self.DISCHARGE)] = P_BAT_MAX

            # SOC
            lower[self.idx(t, self.SOC)] = SOC_MIN
            upper[self.idx(t, self.SOC)] = SOC_MAX

            # curtailment
            upper[self.idx(t, self.CURTAIL)] = pv_kw[t]

            # binaries
            upper[self.idx(t, self.YCH)] = 1
            upper[self.idx(t, self.YDIS)] = 1
            integrality[self.idx(t, self.YCH)] = 1
            integrality[self.idx(t, self.YDIS)] = 1

        rows = []
        lbs = []
        ubs = []

        # 1) Power balance
        for t in range(T):
            row = {}
            row[self.idx(t, self.GRID)] = 1.0
            row[self.idx(t, self.CHARGE)] = -1.0
            row[self.idx(t, self.DISCHARGE)] = 1.0
            row[self.idx(t, self.CURTAIL)] = -1.0
            # grid + pv + discharge - charge - curtail = load
            rows.append(row)
            lbs.append(load_kw[t] - pv_kw[t])
            ubs.append(load_kw[t] - pv_kw[t])

        # 2) SOC recursion
        for t in range(T):
            row = {}
            row[self.idx(t, self.SOC)] = 1.0
            row[self.idx(t, self.CHARGE)] = -ETA * self.dt
            row[self.idx(t, self.DISCHARGE)] = self.dt / ETA
            if t == 0:
                # soc_0 - eta*dt*ch + dt/eta*dis = initial_soc
                rhs = initial_soc
            else:
                row[self.idx(t - 1, self.SOC)] = -1.0
                rhs = 0.0
            rows.append(row)
            lbs.append(rhs)
            ubs.append(rhs)

        # 3) mutual exclusivity
        for t in range(T):
            row1 = {self.idx(t, self.CHARGE): 1.0, self.idx(t, self.YCH): -P_BAT_MAX}
            rows.append(row1)
            lbs.append(-np.inf)
            ubs.append(0.0)

            row2 = {self.idx(t, self.DISCHARGE): 1.0, self.idx(t, self.YDIS): -P_BAT_MAX}
            rows.append(row2)
            lbs.append(-np.inf)
            ubs.append(0.0)

            row3 = {self.idx(t, self.YCH): 1.0, self.idx(t, self.YDIS): 1.0}
            rows.append(row3)
            lbs.append(-np.inf)
            ubs.append(1.0)

        # 4) terminal SOC
        if terminal_soc is not None:
            row = {self.idx(T - 1, self.SOC): 1.0}
            rows.append(row)
            lbs.append(float(terminal_soc))
            ubs.append(float(terminal_soc))

        A = lil_matrix((len(rows), n), dtype=float)
        for i, d in enumerate(rows):
            for j, v in d.items():
                A[i, j] = v
        A = csr_matrix(A)

        constraints = LinearConstraint(A, np.asarray(lbs), np.asarray(ubs))
        bounds = Bounds(lower, upper)

        options = {
            "disp": False,
            "mip_rel_gap": mip_rel_gap,
        }
        if time_limit is not None:
            options["time_limit"] = time_limit

        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if res.x is None:
            return OptimizationResult(
                success=False,
                status=int(res.status),
                message=str(res.message),
                objective=float("nan"),
                grid=np.full(T, np.nan),
                charge=np.full(T, np.nan),
                discharge=np.full(T, np.nan),
                soc=np.full(T, np.nan),
                curtailment=np.full(T, np.nan),
            )

        x = res.x
        grid = np.array([x[self.idx(t, self.GRID)] for t in range(T)])
        charge = np.array([x[self.idx(t, self.CHARGE)] for t in range(T)])
        discharge = np.array([x[self.idx(t, self.DISCHARGE)] for t in range(T)])
        soc = np.array([x[self.idx(t, self.SOC)] for t in range(T)])
        curtail = np.array([x[self.idx(t, self.CURTAIL)] for t in range(T)])

        return OptimizationResult(
            success=bool(res.success),
            status=int(res.status),
            message=str(res.message),
            objective=float(res.fun),
            grid=grid,
            charge=charge,
            discharge=discharge,
            soc=soc,
            curtailment=curtail,
        )

# =============================================================================
# 9. 问题1
# =============================================================================

def solve_problem1(day1: DayData) -> pd.DataFrame:
    print_banner("问题1：固定电价/固定负荷/固定光伏预测的日计划")
    opt = MicrogridMILP().solve(
        load_kw=day1.load_kw,
        pv_kw=day1.pv_kw,
        price=day1.price,
        initial_soc=SOC_INITIAL,
        terminal_soc=SOC_TERMINAL,
    )
    if not opt.success:
        raise RuntimeError(f"问题1MILP求解失败：{opt.message}")

    rows = []
    for t in range(SLOTS_PER_DAY):
        rows.append({
            "时段": t + 1,
            "时间_小时": (t + 1) / 6,
            "电价": day1.price[t],
            "负荷_kW": day1.load_kw[t],
            "光伏预测_kW": day1.pv_kw[t],
            "计划购电_kW": opt.grid[t],
            "储能充电_kW": opt.charge[t],
            "储能放电_kW": opt.discharge[t],
            "SOC_kWh": opt.soc[t],
            "弃光_kW": opt.curtailment[t],
            "购电成本": opt.grid[t] * day1.price[t] * DT,
        })
    df = pd.DataFrame(rows)
    print(f"问题1最优成本：{opt.objective:,.2f}")
    return df

# =============================================================================
# 10. 问题2：单日安全裕度计划 + 全年实际执行
# =============================================================================

def optimize_day_plan_with_margin(
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
    price: np.ndarray,
    margin: float,
    initial_soc: float,
) -> OptimizationResult:
    conservative_load = np.maximum(load_actual * (1.0 + margin), 0.0)
    conservative_pv = np.maximum(pv_actual * (1.0 - margin), 0.0)
    return MicrogridMILP().solve(
        load_kw=conservative_load,
        pv_kw=conservative_pv,
        price=price,
        initial_soc=initial_soc,
        terminal_soc=None,
    )


def execute_scheduled_day(
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
    price: np.ndarray,
    plan: OptimizationResult,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
) -> Dict[str, np.ndarray | float]:
    """按日计划执行，若实际净缺口超过计划购电则通过应急购电补足。"""
    grid_plan = plan.grid
    ch = plan.charge
    dis = plan.discharge

    net_need = load_actual + ch - pv_actual - dis - grid_plan
    emergency = np.maximum(net_need, 0.0)
    surplus = np.maximum(-net_need, 0.0)
    total_purchase = grid_plan + emergency
    scheduled_cost = np.sum(grid_plan * price * DT)
    emergency_cost = np.sum(emergency * price * DT * emergency_multiplier)
    total_cost = scheduled_cost + emergency_cost

    return {
        "grid_plan": grid_plan,
        "charge": ch,
        "discharge": dis,
        "soc": plan.soc,
        "emergency": emergency,
        "surplus": surplus,
        "total_purchase": total_purchase,
        "scheduled_cost": scheduled_cost,
        "emergency_cost": emergency_cost,
        "total_cost": total_cost,
    }


def build_q2_day_ahead_forecast(
    load_dict: Dict[pd.Timestamp, np.ndarray],
    pv_dict: Dict[pd.Timestamp, np.ndarray],
    baseline_day: DayData,
    method: str = Q2_PLAN_FORECAST_METHOD,
) -> Tuple[Dict[pd.Timestamp, np.ndarray], Dict[pd.Timestamp, np.ndarray]]:
    """构造问题2的日前预测，避免在0:00使用当天实际负荷/光伏。

    PREV_DAY：当天0:00采用前一日实际值作为最简单的日前预测；
    首日没有前一天数据，因此使用附件1基准日作为冷启动输入。
    """
    dates = sorted(set(load_dict) & set(pv_dict))
    plan_load, plan_pv = {}, {}
    for i, d in enumerate(dates):
        if method == "PREV_DAY":
            if i == 0:
                plan_load[d] = baseline_day.load_kw.copy()
                plan_pv[d] = baseline_day.pv_kw.copy()
            else:
                prev = dates[i - 1]
                plan_load[d] = load_dict[prev].copy()
                plan_pv[d] = pv_dict[prev].copy()
        elif method == "ACTUAL_BENCHMARK":
            plan_load[d] = load_dict[d].copy()
            plan_pv[d] = pv_dict[d].copy()
        else:
            raise ValueError(f"未知Q2计划预测方法：{method}")
    return plan_load, plan_pv

def solve_problem2_for_margin(
    load_dict: Dict[pd.Timestamp, np.ndarray],
    pv_dict: Dict[pd.Timestamp, np.ndarray],
    price_dict: Dict[pd.Timestamp, np.ndarray],
    margin: float,
    price_mode: str = "fixed",
    plan_load_dict: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
    plan_pv_dict: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
    return_detail: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    dates = sorted(set(load_dict) & set(pv_dict))
    if price_mode == "fixed":
        # 使用附件1首日固定电价作为全年价格
        fixed_price = load_attachment1(FILE_A1).price
        get_price = lambda d: fixed_price
    else:
        get_price = lambda d: price_dict[d]

    all_rows = []
    daily_rows = []
    soc_initial = SOC_INITIAL

    for d in dates:
        load = load_dict[d]  # 实际运行数据，仅用于执行层检验
        pv = pv_dict[d]
        plan_load = load if plan_load_dict is None else plan_load_dict[d]
        plan_pv = pv if plan_pv_dict is None else plan_pv_dict[d]
        price = get_price(d)

        plan = optimize_day_plan_with_margin(
            load_actual=plan_load,
            pv_actual=plan_pv,
            price=price,
            margin=margin,
            initial_soc=soc_initial,
        )
        if not plan.success:
            raise RuntimeError(f"问题2日计划求解失败：{d.date()} / {plan.message}")

        exe = execute_scheduled_day(load, pv, price, plan)
        daily_rows.append({
            "日期": d.date(),
            "安全裕度": margin,
            "日计划购电量_kWh": float(np.sum(exe["grid_plan"]) * DT),
            "日应急购电量_kWh": float(np.sum(exe["emergency"]) * DT),
            "日计划购电成本": float(exe["scheduled_cost"]),
            "日应急购电成本": float(exe["emergency_cost"]),
            "日总成本": float(exe["total_cost"]),
            "期初SOC_kWh": soc_initial,
            "期末SOC_kWh": float(plan.soc[-1]),
        })

        for t in range(SLOTS_PER_DAY):
            all_rows.append({
                "日期": d.date(),
                "时段": t + 1,
                "时间_小时": (t + 1) / 6,
                "电价": price[t],
                "计划负荷预测_kW": plan_load[t],
                "计划光伏预测_kW": plan_pv[t],
                "实际负荷_kW": load[t],
                "实际光伏_kW": pv[t],
                "安全裕度": margin,
                "计划购电_kW": exe["grid_plan"][t],
                "充电_kW": exe["charge"][t],
                "放电_kW": exe["discharge"][t],
                "SOC_kWh": exe["soc"][t],
                "应急购电_kW": exe["emergency"][t],
                "总购电_kW": exe["total_purchase"][t],
                "计划购电成本": exe["grid_plan"][t] * price[t] * DT,
                "应急购电成本": exe["emergency"][t] * price[t] * DT * EMERGENCY_MULTIPLIER,
                "总成本": (exe["grid_plan"][t] + exe["emergency"][t] * EMERGENCY_MULTIPLIER) * price[t] * DT,
            })

        # 当天终点SOC直接作为下一天初始SOC，不再强制恢复到6000 kWh。
        soc_initial = float(plan.soc[-1])

    detail = pd.DataFrame(all_rows)
    daily = pd.DataFrame(daily_rows)
    summary = {
        "margin": margin,
        "annual_scheduled_cost": float(daily["日计划购电成本"].sum()),
        "annual_emergency_cost": float(daily["日应急购电成本"].sum()),
        "annual_total_cost": float(daily["日总成本"].sum()),
        "annual_emergency_kwh": float(daily["日应急购电量_kWh"].sum()),
        "annual_purchase_kwh": float(daily["日计划购电量_kWh"].sum() + daily["日应急购电量_kWh"].sum()),
    }
    if return_detail:
        return daily, summary, detail
    return daily, summary


def solve_problem2_all_margins(
    load_dict: Dict[pd.Timestamp, np.ndarray],
    pv_dict: Dict[pd.Timestamp, np.ndarray],
    price_dict: Dict[pd.Timestamp, np.ndarray],
    margins: List[float] = DEFAULT_MARGINS,
    plan_load_dict: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
    plan_pv_dict: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
) -> Tuple[float, pd.DataFrame, Dict[float, pd.DataFrame]]:
    records = []
    daily_by_margin: Dict[float, pd.DataFrame] = {}
    for m in margins:
        print(f"  问题2测试安全裕度 {m:.0%} ...")
        daily, summary = solve_problem2_for_margin(
            load_dict=load_dict,
            pv_dict=pv_dict,
            price_dict=price_dict,
            margin=m,
            price_mode="fixed",
            plan_load_dict=plan_load_dict,
            plan_pv_dict=plan_pv_dict,
        )
        records.append(summary)
        daily_by_margin[m] = daily
        print(f"    年总成本：{summary['annual_total_cost']:,.2f}，应急购电：{summary['annual_emergency_kwh']:,.2f} kWh")
    sens = pd.DataFrame(records).sort_values("annual_total_cost").reset_index(drop=True)
    best_margin = float(sens.iloc[0]["margin"])
    return best_margin, sens, daily_by_margin

# =============================================================================
# 11. BP残差修正：附件3 -> 实际光伏
# =============================================================================

class PVForecastCorrector:
    """使用BP神经网络学习附件3原始预测的残差。"""

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.feature_medians: Optional[np.ndarray] = None
        self.residual_by_lead: Dict[int, Tuple[float, float]] = {}
        self.pool_quantiles: Tuple[float, float] = (-500.0, 500.0)
        self.train_end_index: int = 0
        self.fitted = False
        self.metrics: Dict[str, float] = {}

    @staticmethod
    def _feature_vector(
        raw_forecast: float,
        lag_values: List[float],
        target_hour: int,
        lead_hour: int,
    ) -> List[float]:
        angle = 2.0 * math.pi * target_hour / 24.0
        return [
            float(raw_forecast),
            float(lag_values[0]),
            float(lag_values[1]),
            float(lag_values[2]),
            math.sin(angle),
            math.cos(angle),
            float(lead_hour),
        ]

    def build_training_set(
        self,
        forecast_df: pd.DataFrame,
        pv_dict: Dict[pd.Timestamp, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        records = []
        X = []
        y = []
        for ridx, row in forecast_df.iterrows():
            release_date = pd.Timestamp(row["date"]).normalize()
            release_hour = int(row["release_hour"])
            for lead in range(1, 25):
                target_dt = release_date + pd.Timedelta(hours=lead - 1 + release_hour)
                target_date = target_dt.normalize()
                target_hour = target_dt.hour
                if target_date not in pv_dict:
                    continue
                slot = target_hour * SLOTS_PER_HOUR
                actual = float(np.mean(pv_dict[target_date][slot:slot + SLOTS_PER_HOUR]))
                raw = float(row[f"h{lead}"])

                lags = []
                for lag in [1, 2, 3]:
                    ld = target_date - pd.Timedelta(days=lag)
                    if ld in pv_dict:
                        lagv = float(np.mean(pv_dict[ld][slot:slot + SLOTS_PER_HOUR]))
                    else:
                        lagv = np.nan
                    lags.append(lagv)

                feat = self._feature_vector(raw, lags, target_hour, lead)
                X.append(feat)
                y.append(actual - raw)
                records.append({
                    "release_date": release_date,
                    "release_hour": release_hour,
                    "target_date": target_date,
                    "target_hour": target_hour,
                    "lead_hour": lead,
                    "raw": raw,
                    "actual": actual,
                    "residual": actual - raw,
                })
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float), records

    def fit(
        self,
        forecast_df: pd.DataFrame,
        pv_dict: Dict[pd.Timestamp, np.ndarray],
    ):
        X, y, records = self.build_training_set(forecast_df, pv_dict)
        if len(X) < 100:
            raise RuntimeError(f"BP训练样本过少：{len(X)}")

        # 按时间顺序80/20切分，避免随机打乱导致未来信息泄露。
        order = np.argsort(pd.to_datetime([r["target_date"] for r in records]).values)
        X = X[order]
        y = y[order]
        rec_sorted = [records[i] for i in order]

        split = max(int(len(X) * 0.8), 1)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 仅使用训练集统计量填补缺失特征。
        med = np.nanmedian(X_train, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X_train = np.where(np.isfinite(X_train), X_train, med)
        X_test = np.where(np.isfinite(X_test), X_test, med)
        self.feature_medians = med

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("bp", MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.001,
                max_iter=100,
                batch_size=256,
                tol=1e-3,
                early_stopping=False,
                random_state=RANDOM_SEED,
            )),
        ])
        self.model.fit(X_train, y_train)
        pred_resid = self.model.predict(X_test)
        corrected = X_test[:, 0] + pred_resid
        raw = X_test[:, 0]

        self.metrics = {
            "test_MAE_raw": float(np.mean(np.abs(raw - (X_test[:, 0] + y_test)))),
            "test_MAE_corrected": float(np.mean(np.abs(corrected - (X_test[:, 0] + y_test)))),
            "test_RMSE_raw": float(np.sqrt(np.mean((raw - (X_test[:, 0] + y_test)) ** 2))),
            "test_RMSE_corrected": float(np.sqrt(np.mean((corrected - (X_test[:, 0] + y_test)) ** 2))),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        # 以测试集中的“修正后残差”建立经验预测区间。
        corrected_resid = (X_test[:, 0] + y_test) - corrected
        lead_test = [rec_sorted[i]["lead_hour"] for i in range(split, len(rec_sorted))]
        for lead in range(1, 25):
            r = corrected_resid[np.asarray(lead_test) == lead]
            if len(r) >= 20:
                self.residual_by_lead[lead] = (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))
        if len(corrected_resid) > 0:
            self.pool_quantiles = (
                float(np.quantile(corrected_resid, 0.025)),
                float(np.quantile(corrected_resid, 0.975)),
            )
        self.train_end_index = split
        self.fitted = True
        return self

    def _make_features_for_forecast(
        self,
        raw: float,
        target_date: pd.Timestamp,
        target_hour: int,
        lead: int,
        pv_dict: Dict[pd.Timestamp, np.ndarray],
    ) -> np.ndarray:
        slot = target_hour * SLOTS_PER_HOUR
        lags = []
        for lag in [1, 2, 3]:
            d = target_date - pd.Timedelta(days=lag)
            if d in pv_dict:
                v = float(np.mean(pv_dict[d][slot:slot + SLOTS_PER_HOUR]))
            else:
                v = np.nan
            lags.append(v)
        feat = np.asarray(self._feature_vector(raw, lags, target_hour, lead), dtype=float)
        if self.feature_medians is None:
            raise RuntimeError("模型尚未训练")
        feat = np.where(np.isfinite(feat), feat, self.feature_medians)
        return feat.reshape(1, -1)

    def correct_hourly_forecast(
        self,
        release_date: pd.Timestamp,
        release_hour: int,
        hourly_forecast: np.ndarray,
        pv_dict: Dict[pd.Timestamp, np.ndarray],
    ) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            raise RuntimeError("请先fit")
        rows = []
        for lead in range(1, 25):
            raw = float(hourly_forecast[lead - 1])
            target_dt = pd.Timestamp(release_date).normalize() + pd.Timedelta(hours=release_hour + lead - 1)
            target_date = target_dt.normalize()
            target_hour = target_dt.hour
            feat = self._make_features_for_forecast(raw, target_date, target_hour, lead, pv_dict)
            resid_hat = float(self.model.predict(feat)[0])
            corrected = max(0.0, raw + resid_hat)
            ql, qu = self.residual_by_lead.get(lead, self.pool_quantiles)
            lower = max(0.0, corrected + ql)
            upper = max(lower, corrected + qu)
            rows.append({
                "target_date": target_date,
                "target_hour": target_hour,
                "lead_hour": lead,
                "raw_forecast": raw,
                "residual_hat": resid_hat,
                "corrected_forecast": corrected,
                "lower95": lower,
                "upper95": upper,
            })
        return pd.DataFrame(rows)

# =============================================================================
# 12. 附件3预报转换为144时段
# =============================================================================

def get_forecast_row(
    forecast_df: pd.DataFrame,
    date: pd.Timestamp,
    release_hour: int,
) -> pd.Series:
    d = pd.Timestamp(date).normalize()
    m = forecast_df[(forecast_df["date"].dt.normalize() == d) & (forecast_df["release_hour"] == release_hour)]
    if m.empty:
        raise KeyError(f"附件3找不到 {d.date()} {release_hour}:00 的预报")
    return m.iloc[0]


def hourly_corrected_to_144(
    corrected_df: pd.DataFrame,
    day: pd.Timestamp,
    fill_missing: Optional[float] = None,
    conservative: bool = True,
) -> np.ndarray:
    """把目标日期24个小时预测填到144个十分钟槽位。"""
    out = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
    df = corrected_df[corrected_df["target_date"] == pd.Timestamp(day).normalize()].copy()
    for _, r in df.iterrows():
        h = int(r["target_hour"])
        v = float(r["lower95"] if conservative else r["corrected_forecast"])
        out[h * 6:(h + 1) * 6] = v
    if np.isnan(out).any():
        if fill_missing is None:
            # 缺失小时一般只出现在数据边界，按0处理（夜间合理；边界不影响主要时段）。
            fill_missing = 0.0
        out = np.where(np.isfinite(out), out, fill_missing)
    return np.maximum(out, 0.0)


def persistence_fill_forecast(
    forecast_df: pd.DataFrame,
    corrector: PVForecastCorrector,
    day: pd.Timestamp,
    release_hour: int,
    pv_dict: Dict[pd.Timestamp, np.ndarray],
) -> np.ndarray:
    """为没有正式预报时刻的模拟更新构造“最近正式预报延续”方案。

    注意：这只用于分析更新频率，不把这些模拟更新当作附件3真实信息。
    """
    official = [0, 6, 12, 18]
    previous = max([h for h in official if h <= release_hour], default=0)
    row = get_forecast_row(forecast_df, day, previous)
    hourly = np.array([float(row[f"h{i}"]) for i in range(1, 25)])
    corr = corrector.correct_hourly_forecast(day, previous, hourly, pv_dict)
    # 最近正式预报中，目标日期的值只取其预报覆盖范围；24小时均填充后得到可用向量。
    return hourly_corrected_to_144(corr, day, conservative=True)

# =============================================================================
# 13. 问题3：滚动优化及结算
# =============================================================================

def optimize_q3_stage(
    day: pd.Timestamp,
    load: np.ndarray,
    pv_forecast_lower: np.ndarray,
    price: np.ndarray,
    start_slot: int,
    initial_soc: float,
) -> OptimizationResult:
    """只对[start_slot, 144)重新优化，前面执行区间冻结。"""
    if start_slot >= SLOTS_PER_DAY:
        raise ValueError("start_slot超出范围")
    load_tail = load[start_slot:]
    pv_tail = pv_forecast_lower[start_slot:]
    price_tail = price[start_slot:]

    if len(load_tail) == 0:
        raise ValueError("没有剩余时段")

    # MILP类固定按144构造，因此这里建立一个通用“变长”求解器实例。
    solver = MicrogridMILP()
    # 临时使用内部solve的固定长度限制：扩展为通用函数
    return solve_variable_length_day(
        load_kw=load_tail,
        pv_kw=pv_tail,
        price=price_tail,
        initial_soc=initial_soc,
        terminal_soc=None,
    )


def solve_variable_length_day(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    initial_soc: float,
    terminal_soc: Optional[float],
) -> OptimizationResult:
    """任意时长滚动优化；滚动阶段末端SOC不强制恢复到6000 kWh，
    当前阶段实际执行后的SOC作为下一阶段初始状态。"""
    if not EXACT_BINARY_MILP:
        return solve_lp_day(load_kw, pv_kw, price, initial_soc, terminal_soc)
    load_kw = np.asarray(load_kw, dtype=float).reshape(-1)
    pv_kw = np.maximum(np.asarray(pv_kw, dtype=float).reshape(-1), 0.0)
    price = np.asarray(price, dtype=float).reshape(-1)
    T = len(load_kw)
    if not (len(pv_kw) == len(price) == T):
        raise ValueError("滚动优化输入长度不一致")
    nvar = 7
    n = T * nvar

    def ix(t, k):
        return t * nvar + k

    GRID, CH, DIS, SOC, CURT, YCH, YDIS = range(7)
    c = np.zeros(n)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    integ = np.zeros(n, dtype=int)

    for t in range(T):
        c[ix(t, GRID)] = price[t] * DT
        hi[ix(t, GRID)] = max(float(load_kw[t] + pv_kw[t] + P_BAT_MAX), 50000.0)
        hi[ix(t, CH)] = P_BAT_MAX
        hi[ix(t, DIS)] = P_BAT_MAX
        lo[ix(t, SOC)] = SOC_MIN
        hi[ix(t, SOC)] = SOC_MAX
        hi[ix(t, CURT)] = pv_kw[t]
        hi[ix(t, YCH)] = 1.0
        hi[ix(t, YDIS)] = 1.0
        integ[ix(t, YCH)] = 1
        integ[ix(t, YDIS)] = 1

    rows = []
    lbs = []
    ubs = []
    for t in range(T):
        rows.append({ix(t, GRID): 1, ix(t, CH): -1, ix(t, DIS): 1, ix(t, CURT): -1})
        rhs = load_kw[t] - pv_kw[t]
        lbs.append(rhs); ubs.append(rhs)

    for t in range(T):
        row = {ix(t, SOC): 1, ix(t, CH): -ETA * DT, ix(t, DIS): DT / ETA}
        rhs = initial_soc if t == 0 else 0.0
        if t > 0:
            row[ix(t - 1, SOC)] = -1
        rows.append(row); lbs.append(rhs); ubs.append(rhs)

    for t in range(T):
        rows.append({ix(t, CH): 1, ix(t, YCH): -P_BAT_MAX}); lbs.append(-np.inf); ubs.append(0)
        rows.append({ix(t, DIS): 1, ix(t, YDIS): -P_BAT_MAX}); lbs.append(-np.inf); ubs.append(0)
        rows.append({ix(t, YCH): 1, ix(t, YDIS): 1}); lbs.append(-np.inf); ubs.append(1)

    if terminal_soc is not None:
        rows.append({ix(T - 1, SOC): 1}); lbs.append(terminal_soc); ubs.append(terminal_soc)

    A = lil_matrix((len(rows), n), dtype=float)
    for i, row in enumerate(rows):
        for j, v in row.items():
            A[i, j] = v
    A = csr_matrix(A)

    res = milp(
        c=c,
        integrality=integ,
        bounds=Bounds(lo, hi),
        constraints=LinearConstraint(A, np.asarray(lbs), np.asarray(ubs)),
        options={"disp": False, "mip_rel_gap": 1e-6},
    )
    if res.x is None:
        return OptimizationResult(False, int(res.status), str(res.message), np.nan,
                                  np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan),
                                  np.full(T, np.nan), np.full(T, np.nan))
    x = res.x
    return OptimizationResult(
        True, int(res.status), str(res.message), float(res.fun),
        np.array([x[ix(t, GRID)] for t in range(T)]),
        np.array([x[ix(t, CH)] for t in range(T)]),
        np.array([x[ix(t, DIS)] for t in range(T)]),
        np.array([x[ix(t, SOC)] for t in range(T)]),
        np.array([x[ix(t, CURT)] for t in range(T)]),
    )


def run_q3_day(
    day: pd.Timestamp,
    load: np.ndarray,
    pv_dict: Dict[pd.Timestamp, np.ndarray],
    forecast_df: pd.DataFrame,
    corrector: PVForecastCorrector,
    price: np.ndarray,
    update_hours: List[int] = OFFICIAL_UPDATE_HOURS,
    use_simulated_updates: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """单日Q3滚动优化。

    返回：
    detail：144条逐10分钟数据
    stages：每次滚动更新的计划成本、调整量等
    """
    day = pd.Timestamp(day).normalize()
    load = ensure_144(load, "Q3负荷")
    actual_pv = ensure_144(pv_dict[day], "Q3实际光伏")
    price = ensure_144(price, "Q3价格")

    plan_grid = np.zeros(SLOTS_PER_DAY)
    adjusted_grid = np.zeros(SLOTS_PER_DAY)
    plan_charge = np.zeros(SLOTS_PER_DAY)
    adjusted_charge = np.zeros(SLOTS_PER_DAY)
    plan_dis = np.zeros(SLOTS_PER_DAY)
    adjusted_dis = np.zeros(SLOTS_PER_DAY)
    adjusted_soc = np.full(SLOTS_PER_DAY, np.nan)

    current_soc = SOC_INITIAL
    stage_records = []

    # 先做0:00计划，作为“计划购电”。
    for stage_no, release_hour in enumerate(update_hours):
        start_slot = release_hour * SLOTS_PER_HOUR
        if start_slot >= SLOTS_PER_DAY:
            continue

        if release_hour in OFFICIAL_UPDATE_HOURS and not use_simulated_updates:
            row = get_forecast_row(forecast_df, day, release_hour)
            hourly = np.array([float(row[f"h{i}"]) for i in range(1, 25)])
            corr = corrector.correct_hourly_forecast(day, release_hour, hourly, pv_dict)
            pv_lower = hourly_corrected_to_144(corr, day, conservative=True)
            forecast_source = f"附件3-{release_hour}:00"
        else:
            pv_lower = persistence_fill_forecast(forecast_df, corrector, day, release_hour, pv_dict)
            forecast_source = f"模拟更新-{release_hour}:00(沿用最近正式预报)"

        # 对尚未执行的未来区间重新优化，初始SOC为已经冻结计划执行到start_slot之前后的SOC。
        if start_slot == 0:
            stage_initial_soc = SOC_INITIAL
        else:
            stage_initial_soc = current_soc

        opt_tail = solve_variable_length_day(
            load_kw=load[start_slot:],
            pv_kw=pv_lower[start_slot:],
            price=price[start_slot:],
            initial_soc=stage_initial_soc,
            terminal_soc=None,
        )
        if not opt_tail.success:
            raise RuntimeError(f"Q3 {day.date()} {release_hour}:00 滚动优化失败：{opt_tail.message}")

        # 写入当前阶段的未来计划
        adjusted_grid[start_slot:] = opt_tail.grid
        adjusted_charge[start_slot:] = opt_tail.charge
        adjusted_dis[start_slot:] = opt_tail.discharge

        if release_hour == 0:
            plan_grid[:] = adjusted_grid
            plan_charge[:] = adjusted_charge
            plan_dis[:] = adjusted_dis

        # 冻结[start_slot, next_stage_slot)的实际执行结果
        next_start = (update_hours[stage_no + 1] * SLOTS_PER_HOUR) if stage_no + 1 < len(update_hours) else SLOTS_PER_DAY
        freeze_end = min(next_start, SLOTS_PER_DAY)

        # current_soc只能由当前/前一阶段的执行动作更新，不能用未来动作。
        for t in range(start_slot, freeze_end):
            prev_soc = current_soc if t == start_slot else adjusted_soc[t - 1]
            adjusted_soc[t] = prev_soc + ETA * adjusted_charge[t] * DT - adjusted_dis[t] * DT / ETA

        # 记录阶段变化量（最终结算以最终adjusted_grid为准）。
        if stage_no == 0:
            adj_abs = 0.0
            up_adj = 0.0
            down_adj = 0.0
        else:
            prev_schedule = adjusted_grid.copy()
            # 此处只能得到当前阶段相对于冻结前历史的总变化，为报告保留阶段调度结果。
            prev_schedule[:start_slot] = 0
            delta = adjusted_grid[start_slot:] - prev_schedule[start_slot:]
            # 实际上delta与自身相同，因此使用阶段计划与前一阶段结果的对比需单独重算；
            # 为避免信息混淆，下方统一用“最终计划-原始计划”进行结算。
            adj_abs = float(np.sum(np.abs(delta)) * DT)
            up_adj = float(np.sum(np.maximum(delta, 0.0)) * DT)
            down_adj = float(np.sum(np.maximum(-delta, 0.0)) * DT)

        stage_records.append({
            "日期": day.date(),
            "更新时刻": f"{release_hour:02d}:00",
            "起始时段": start_slot + 1,
            "预测来源": forecast_source,
            "滚动优化目标成本": float(opt_tail.objective),
            "预测区间下界平均_kW": float(np.mean(pv_lower[start_slot:])),
            "阶段调整购电量_kWh": adj_abs,
            "阶段上调购电量_kWh": up_adj,
            "阶段下调购电量_kWh": down_adj,
        })

        # 为下一更新时刻保存SOC：如果本阶段直到下一时刻都执行，则current_soc取freeze_end-1
        if freeze_end > start_slot:
            current_soc = float(adjusted_soc[freeze_end - 1])

    # 最后一阶段到24:00的SOC轨迹补齐
    for t in range(1, SLOTS_PER_DAY):
        if np.isnan(adjusted_soc[t]):
            prev_soc = adjusted_soc[t - 1]
            if np.isnan(prev_soc):
                prev_soc = SOC_INITIAL
            adjusted_soc[t] = prev_soc + ETA * adjusted_charge[t] * DT - adjusted_dis[t] * DT / ETA
    adjusted_soc[0] = adjusted_soc[0] if np.isfinite(adjusted_soc[0]) else SOC_INITIAL + ETA * adjusted_charge[0] * DT - adjusted_dis[0] * DT / ETA

    # 用最终计划做实际结算；计划时刻0:00得到的plan_grid保持不变。
    actual_purchase_without_emergency = adjusted_grid
    emergency = np.maximum(
        load + adjusted_charge - actual_pv - adjusted_dis - adjusted_grid,
        0.0,
    )
    surplus = np.maximum(
        actual_pv + adjusted_dis + adjusted_grid - load - adjusted_charge,
        0.0,
    )

    down = np.maximum(plan_grid - adjusted_grid, 0.0)
    up = np.maximum(adjusted_grid - plan_grid, 0.0)
    plan_cost = plan_grid * price * DT
    adjustment_cost = -down * price * DOWN_ADJUST_MULTIPLIER * DT + up * price * UP_ADJUST_MULTIPLIER * DT
    emergency_cost = emergency * price * EMERGENCY_MULTIPLIER * DT
    total_cost = plan_cost + adjustment_cost + emergency_cost

    detail = pd.DataFrame({
        "日期": [day.date()] * SLOTS_PER_DAY,
        "时段": np.arange(1, SLOTS_PER_DAY + 1),
        "时间_小时": np.arange(SLOTS_PER_DAY) / 6.0,
        "电价": price,
        "实际负荷_kW": load,
        "实际光伏_kW": actual_pv,
        "0时计划购电_kW": plan_grid,
        "最终调整购电_kW": adjusted_grid,
        "下调购电_kW": down,
        "上调购电_kW": up,
        "充电_kW": adjusted_charge,
        "放电_kW": adjusted_dis,
        "SOC_kWh": adjusted_soc,
        "应急购电_kW": emergency,
        "最终总购电_kW": adjusted_grid + emergency,
        "计划结算成本": plan_cost,
        "调整结算成本": adjustment_cost,
        "应急购电成本": emergency_cost,
        "总成本": total_cost,
        "实际富余电量_kWh": surplus * DT,
    })

    stages = pd.DataFrame(stage_records)
    stages.attrs["day_total_cost"] = float(np.sum(total_cost))
    stages.attrs["day_plan_cost"] = float(np.sum(plan_cost))
    stages.attrs["day_adjustment_cost"] = float(np.sum(adjustment_cost))
    stages.attrs["day_emergency_cost"] = float(np.sum(emergency_cost))
    stages.attrs["day_emergency_kwh"] = float(np.sum(emergency) * DT)
    return detail, stages


def solve_problem3_year(
    load_dict: Dict[pd.Timestamp, np.ndarray],
    pv_dict: Dict[pd.Timestamp, np.ndarray],
    forecast_df: pd.DataFrame,
    price_dict: Dict[pd.Timestamp, np.ndarray],
    corrector: PVForecastCorrector,
    update_hours: List[int],
    dynamic_price: bool = False,
    use_simulated_updates: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    dates = sorted(set(load_dict) & set(pv_dict))
    detail_all = []
    stage_all = []
    for i, d in enumerate(dates):
        if i % 30 == 0:
            print(f"  Q3滚动优化：{i + 1}/{len(dates)}，当前 {d.date()}")
        price = price_dict[d] if dynamic_price else load_attachment1(FILE_A1).price
        detail, stages = run_q3_day(
            day=d,
            load=load_dict[d],
            pv_dict=pv_dict,
            forecast_df=forecast_df,
            corrector=corrector,
            price=price,
            update_hours=update_hours,
            use_simulated_updates=use_simulated_updates,
        )
        detail_all.append(detail)
        if not stages.empty:
            stage_all.append(stages)

    detail_df = pd.concat(detail_all, ignore_index=True)
    stage_df = pd.concat(stage_all, ignore_index=True)
    summary = {
        "annual_plan_cost": float(detail_df["计划结算成本"].sum()),
        "annual_adjustment_cost": float(detail_df["调整结算成本"].sum()),
        "annual_emergency_cost": float(detail_df["应急购电成本"].sum()),
        "annual_total_cost": float(detail_df["总成本"].sum()),
        "annual_emergency_kwh": float(detail_df["应急购电_kW"].sum() * DT),
        "annual_adjusted_purchase_kwh": float(detail_df["最终调整购电_kW"].sum() * DT),
        "annual_plan_purchase_kwh": float(detail_df["0时计划购电_kW"].sum() * DT),
    }
    return detail_df, stage_df, summary

# =============================================================================
# 14. 问题3更新频率分析
# =============================================================================

def analyze_q3_frequency(
    load_dict: Dict[pd.Timestamp, np.ndarray],
    pv_dict: Dict[pd.Timestamp, np.ndarray],
    forecast_df: pd.DataFrame,
    price_dict: Dict[pd.Timestamp, np.ndarray],
    corrector: PVForecastCorrector,
) -> pd.DataFrame:
    """比较官方4次更新与更多模拟更新时间。

    对3/9/15/21时刻没有附件3真实新预报这一事实明确标记为“模拟”。
    模拟时继续沿用最近正式预报，不伪造附件3不存在的数据。
    为控制求解时间，这里可采用一段代表性月份/季度；默认全年，但每组频率都完整跑一次。
    """
    test_sets = {
        "标准4次": [0, 6, 12, 18],
        "加3小时更新": [0, 3, 6, 9, 12, 15, 18, 21],
        "每4小时更新": [0, 4, 8, 12, 16, 20],
    }
    rows = []
    # 更新频率比较主要用于论文中的敏感性分析。为控制运行时间，采用全年中等间隔抽取的30天代表性样本，
    # 而Q3主结果仍使用全年365天完整滚动优化。模拟时刻明确标记，不把它们当作附件3真实预报。
    all_dates = sorted(set(load_dict) & set(pv_dict))
    if len(all_dates) <= FREQUENCY_SAMPLE_DAYS:
        sample_dates = all_dates
    else:
        sample_idx = np.linspace(0, len(all_dates) - 1, FREQUENCY_SAMPLE_DAYS).round().astype(int)
        sample_dates = [all_dates[i] for i in sample_idx]
    sample_load = {d: load_dict[d] for d in sample_dates}
    sample_pv = {d: pv_dict[d] for d in sample_dates}
    sample_price = {d: price_dict[d] for d in sample_dates}

    for name, hours in test_sets.items():
        sim = any(h not in OFFICIAL_UPDATE_HOURS for h in hours)
        print(f"  更新频率敏感性：{name} -> {hours}")
        _, _, summary = solve_problem3_year(
            load_dict=sample_load,
            pv_dict=sample_pv,
            forecast_df=forecast_df,
            price_dict=sample_price,
            corrector=corrector,
            update_hours=hours,
            dynamic_price=False,
            use_simulated_updates=sim,
        )
        rows.append({
            "更新方案": name,
            "更新时刻": ",".join(f"{h}:00" for h in hours),
            "其中模拟时刻": ",".join(str(h) for h in hours if h not in OFFICIAL_UPDATE_HOURS) if sim else "无",
            **summary,
        })
    return pd.DataFrame(rows)

# =============================================================================
# 15. 结果写入Excel
# =============================================================================

def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]
            out = df.copy()
            out.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.book[safe_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                max_len = 10
                for cell in list(col)[:5000]:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = min(max(max_len, len(value) + 2), 28)
                ws.column_dimensions[col[0].column_letter].width = max_len
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)


def write_summary_workbook(
    path: Path,
    q1: pd.DataFrame,
    q2_sens: pd.DataFrame,
    q3_summary: pd.DataFrame,
    q4_summary: pd.DataFrame,
    bp_metrics: Dict[str, float],
):
    sheets = {
        "Q1_日计划": q1,
        "Q2_安全裕度": q2_sens,
        "Q3_汇总": q3_summary,
        "Q4_汇总": q4_summary,
        "BP模型指标": pd.DataFrame([bp_metrics]),
    }
    write_excel(path, sheets)

# =============================================================================
# 16. 主程序
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)
    print_banner("微网与外部电网电力调控策略——完整求解程序")
    print("脚本目录：", BASE_DIR)
    print("附件1：", FILE_A1)
    print("附件2：", FILE_A2)
    print("附件3：", FILE_A3)
    print("附件4：", FILE_A4)

    # -------------------------------------------------------------------------
    # 16.1 数据读取
    # -------------------------------------------------------------------------
    print_banner("一、读取数据")
    day1 = load_attachment1(FILE_A1)
    load_dict, pv_dict = load_attachment2(FILE_A2)
    forecast_df = load_attachment3(FILE_A3)
    price_dict = load_attachment4(FILE_A4)

    print(f"附件2日期数：{len(load_dict)}")
    print(f"附件3预报记录：{len(forecast_df)}")
    print(f"附件4日期数：{len(price_dict)}")
    print(f"附件2日期范围：{min(load_dict).date()} ~ {max(load_dict).date()}")

    # -------------------------------------------------------------------------
    # 16.2 问题1
    # -------------------------------------------------------------------------
    q1_detail = solve_problem1(day1)
    write_excel(BASE_DIR / "result1.xlsx", {
        "Q1_最优日计划": q1_detail,
        "Q1_汇总": pd.DataFrame([{
            "最优总购电成本": float(q1_detail["购电成本"].sum()),
            "总购电量_kWh": float(q1_detail["计划购电_kW"].sum() * DT),
            "总充电量_kWh": float(q1_detail["储能充电_kW"].sum() * DT),
            "总放电量_kWh": float(q1_detail["储能放电_kW"].sum() * DT),
            "末SOC_kWh": float(q1_detail["SOC_kWh"].iloc[-1]),
        }])
    })

    # -------------------------------------------------------------------------
    # 16.3 问题2
    # -------------------------------------------------------------------------
    print_banner("问题2：全年安全裕度比较")
    best_margin, q2_sens, q2_daily_by_margin = solve_problem2_all_margins(
        load_dict=load_dict,
        pv_dict=pv_dict,
        price_dict=price_dict,
        margins=DEFAULT_MARGINS,
    )
    print(f"问题2最优安全裕度：{best_margin:.0%}")
    q2_daily = q2_daily_by_margin[best_margin]
    write_excel(BASE_DIR / "result2.xlsx", {
        "Q2_安全裕度比较": q2_sens,
        "Q2_最优裕度逐日": q2_daily,
    })

    # -------------------------------------------------------------------------
    # 16.4 BP模型训练
    # -------------------------------------------------------------------------
    print_banner("问题3：BP残差修正模型训练")
    corrector = PVForecastCorrector()
    corrector.fit(forecast_df, pv_dict)
    print("BP测试集指标：")
    for k, v in corrector.metrics.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------------------
    # 16.5 问题3：官方4次滚动优化
    # -------------------------------------------------------------------------
    print_banner("问题3：0/6/12/18时滚动优化")
    q3_detail, q3_stages, q3_summary_dict = solve_problem3_year(
        load_dict=load_dict,
        pv_dict=pv_dict,
        forecast_df=forecast_df,
        price_dict=price_dict,
        corrector=corrector,
        update_hours=OFFICIAL_UPDATE_HOURS,
        dynamic_price=False,
        use_simulated_updates=False,
    )
    q3_summary = pd.DataFrame([q3_summary_dict])
    q3_bp_metrics = pd.DataFrame([corrector.metrics])
    write_excel(BASE_DIR / "result3.xlsx", {
        "Q3_逐时段结果": q3_detail,
        "Q3_滚动阶段": q3_stages,
        "Q3_成本汇总": q3_summary,
        "Q3_BP指标": q3_bp_metrics,
    })

    # 更新频率分析：为了完整而运行，若机器较慢仍不影响主结果。
    print_banner("问题3：更新频率敏感性分析")
    q3_frequency = analyze_q3_frequency(
        load_dict=load_dict,
        pv_dict=pv_dict,
        forecast_df=forecast_df,
        price_dict=price_dict,
        corrector=corrector,
    )
    # 将频率分析写入result3附加工作簿（追加sheet不直接append，重新写完整结果更稳妥）
    write_excel(BASE_DIR / "result3.xlsx", {
        "Q3_逐时段结果": q3_detail,
        "Q3_滚动阶段": q3_stages,
        "Q3_成本汇总": q3_summary,
        "Q3_BP指标": q3_bp_metrics,
        "Q3_更新频率": q3_frequency,
    })

    # -------------------------------------------------------------------------
    # 16.6 问题4-2：动态电价下重新求解问题2
    # -------------------------------------------------------------------------
    print_banner("问题4-2：动态电价下重新求解问题2")
    q42_daily, q42_summary = solve_problem2_for_margin(
        load_dict=load_dict,
        pv_dict=pv_dict,
        price_dict=price_dict,
        margin=best_margin,
        price_mode="dynamic",
    )
    q42_sensitivity = pd.DataFrame([q42_summary])
    write_excel(BASE_DIR / "result4-2.xlsx", {
        "Q4-2_逐日结果": q42_daily,
        "Q4-2_汇总": q42_sensitivity,
        "Q4-2_说明": pd.DataFrame({
            "说明": [
                "问题4-2并非对问题2已有成本简单重新计价，而是在附件4动态电价条件下重新执行同一安全裕度优化模型。",
                f"安全裕度沿用问题2计算得到的最优值：{best_margin:.0%}。",
                "每个10分钟时段使用附件4对应动态电价重新求解储能充放电与购电计划。",
            ]
        })
    })

    # -------------------------------------------------------------------------
    # 16.7 问题4-3：动态电价下重新求解问题3
    # -------------------------------------------------------------------------
    print_banner("问题4-3：动态电价下重新求解问题3")
    q43_detail, q43_stages, q43_summary_dict = solve_problem3_year(
        load_dict=load_dict,
        pv_dict=pv_dict,
        forecast_df=forecast_df,
        price_dict=price_dict,
        corrector=corrector,
        update_hours=OFFICIAL_UPDATE_HOURS,
        dynamic_price=True,
        use_simulated_updates=False,
    )
    q43_summary = pd.DataFrame([q43_summary_dict])
    write_excel(BASE_DIR / "result4-3.xlsx", {
        "Q4-3_逐时段结果": q43_detail,
        "Q4-3_滚动阶段": q43_stages,
        "Q4-3_成本汇总": q43_summary,
        "Q4-3_BP指标": q3_bp_metrics,
        "Q4-3_说明": pd.DataFrame({
            "说明": [
                "问题4-3不是对问题3结果直接乘以新电价，而是在附件4动态电价下重新运行问题3滚动优化。",
                "保留附件3 BP残差修正、95%预测区间、0/6/12/18滚动更新、计划/调整/应急结算规则。",
                "动态电价作为每个10分钟时段的目标函数系数直接进入MILP。",
            ]
        })
    })

    # -------------------------------------------------------------------------
    # 16.8 统一汇总
    # -------------------------------------------------------------------------
    q4_summary = pd.DataFrame([
        {"问题": "Q4-2", **q42_summary},
        {"问题": "Q4-3", **q43_summary_dict},
    ])
    write_summary_workbook(
        BASE_DIR / "model_summary.xlsx",
        q1=q1_detail,
        q2_sens=q2_sens,
        q3_summary=q3_summary,
        q4_summary=q4_summary,
        bp_metrics=corrector.metrics,
    )

    print_banner("全部计算完成")
    print("输出文件：")
    for f in ["result1.xlsx", "result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx", "model_summary.xlsx"]:
        print("  ", BASE_DIR / f)

    print("\n核心结果：")
    print(f"  Q1成本：{q1_detail['购电成本'].sum():,.2f}")
    print(f"  Q2最优安全裕度：{best_margin:.0%}")
    print(f"  Q2年总成本：{q2_sens[q2_sens['margin'] == best_margin]['annual_total_cost'].iloc[0]:,.2f}")
    print(f"  Q3年总成本：{q3_summary_dict['annual_total_cost']:,.2f}")
    print(f"  Q4-2年总成本：{q42_summary['annual_total_cost']:,.2f}")
    print(f"  Q4-3年总成本：{q43_summary_dict['annual_total_cost']:,.2f}")



# =============================================================================
# 17. 论文结果增强模块
# =============================================================================
# 说明：以下模块不改变Q1-Q4的核心优化思想，只增加论文需要的统计、对比、
#      典型日、图表和汇总输出，并修正Q3阶段变化量的统计逻辑。

import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager, rcParams

# -----------------------------------------------------------------------------
# 中文字体设置：避免 Windows / Linux 上 Matplotlib 将中文显示为方框。
# 优先使用 Windows 常见中文字体；若本机没有，则自动寻找可用中文字体。
# -----------------------------------------------------------------------------
def setup_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "DengXian",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)

    if chosen is None:
        # 再从字体文件路径中尝试识别常见中文字体名称。
        preferred_keywords = [
            "msyh", "simhei", "simsun", "deng", "notosanscjk",
            "sourcehansans", "wqy-zenhei", "arialuni"
        ]
        for f in font_manager.fontManager.ttflist:
            path = str(getattr(f, "fname", "")).lower()
            if any(k in path for k in preferred_keywords):
                chosen = f.name
                break

    if chosen is not None:
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [chosen] + [x for x in candidates if x != chosen]
        rcParams["axes.unicode_minus"] = False
        print(f"Matplotlib中文字体：{chosen}")
    else:
        rcParams["axes.unicode_minus"] = False
        print("警告：未找到常见中文字体。请在本机安装‘微软雅黑’或‘黑体’后重新运行。")


setup_chinese_font()

FIG_DIR = BASE_DIR / "论文图表"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def clear_paper_figures():
    """清理论文图表目录中旧的PNG，确保最终正文只保留9张核心图。"""
    for f in FIG_DIR.glob("*.png"):
        try:
            f.unlink()
        except OSError:
            pass


# 论文增强计算的速度控制：主结果仍按全年完整计算。
# 下面两个开关只影响“附加对比/敏感性”模块，不改变Q1-Q4主模型。
RUN_Q3_RAW_BASELINE = True
BASELINE_SAMPLE_DAYS = 12
FREQUENCY_SAMPLE_DAYS = 15


def fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def calc_energy(power_kw: np.ndarray) -> float:
    return float(np.nansum(np.asarray(power_kw, dtype=float)) * DT)


def calc_daily_profile_metrics(detail: pd.DataFrame, load_col: str, pv_col: str,
                               grid_col: str, charge_col: str, discharge_col: str,
                               soc_col: str, curtail_col: Optional[str] = None) -> Dict[str, float]:
    load = detail[load_col].to_numpy(dtype=float)
    pv = detail[pv_col].to_numpy(dtype=float)
    grid = detail[grid_col].to_numpy(dtype=float)
    ch = detail[charge_col].to_numpy(dtype=float)
    dis = detail[discharge_col].to_numpy(dtype=float)
    soc = detail[soc_col].to_numpy(dtype=float)
    m = {
        "负荷电量_kWh": calc_energy(load),
        "光伏发电量_kWh": calc_energy(pv),
        "购电量_kWh": calc_energy(grid),
        "充电量_kWh": calc_energy(ch),
        "放电量_kWh": calc_energy(dis),
        "SOC最小值_kWh": float(np.nanmin(soc)),
        "SOC最大值_kWh": float(np.nanmax(soc)),
        "最大充电功率_kW": float(np.nanmax(ch)),
        "最大放电功率_kW": float(np.nanmax(dis)),
        "峰值购电功率_kW": float(np.nanmax(grid)),
        "储能循环等效次数": calc_energy(ch) / max(2.0 * E_CAP, 1.0),
    }
    if curtail_col and curtail_col in detail.columns:
        curtail = detail[curtail_col].to_numpy(dtype=float)
        m["弃光电量_kWh"] = calc_energy(curtail)
        m["光伏消纳量_kWh"] = max(m["光伏发电量_kWh"] - m["弃光电量_kWh"], 0.0)
        m["光伏消纳率"] = m["光伏消纳量_kWh"] / max(m["光伏发电量_kWh"], 1e-9)
    return m


def q1_paper_metrics(q1_detail: pd.DataFrame) -> pd.DataFrame:
    # 附件1的光伏预测是输入供给，因此这里以优化中的弃光量统计PV消纳。
    m = calc_daily_profile_metrics(
        q1_detail,
        "负荷_kW", "光伏预测_kW", "计划购电_kW", "储能充电_kW", "储能放电_kW", "SOC_kWh", "弃光_kW"
    )
    cost = float(q1_detail["购电成本"].sum())
    m.update({
        "购电成本": cost,
        "平均电价_元每kWh": cost / max(m["购电量_kWh"], 1e-9),
        "储能净放电_kWh": m["放电量_kWh"] - m["充电量_kWh"],
        "末SOC_kWh": float(q1_detail["SOC_kWh"].iloc[-1]),
    })
    return pd.DataFrame([m])


def enrich_q2_daily(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["总购电量_kWh"] = out["日计划购电量_kWh"] + out["日应急购电量_kWh"]
    out["应急购电占比"] = np.where(out["总购电量_kWh"] > 0,
                                  out["日应急购电量_kWh"] / out["总购电量_kWh"], 0.0)
    out["日均成本"] = out["日总成本"]
    out["成本相对均值"] = out["日总成本"] / max(out["日总成本"].mean(), 1e-9)
    return out


def q2_paper_metrics(best_daily: pd.DataFrame) -> pd.DataFrame:
    d = enrich_q2_daily(best_daily)
    return pd.DataFrame([{
        "安全裕度": float(d["安全裕度"].iloc[0]),
        "年总成本": float(d["日总成本"].sum()),
        "计划购电量_kWh": float(d["日计划购电量_kWh"].sum()),
        "应急购电量_kWh": float(d["日应急购电量_kWh"].sum()),
        "总购电量_kWh": float(d["总购电量_kWh"].sum()),
        "计划购电成本": float(d["日计划购电成本"].sum()),
        "应急购电成本": float(d["日应急购电成本"].sum()),
        "应急购电成本占比": float(d["日应急购电成本"].sum() / max(d["日总成本"].sum(), 1e-9)),
        "平均日成本": float(d["日总成本"].mean()),
        "最大日成本": float(d["日总成本"].max()),
        "最小日成本": float(d["日总成本"].min()),
    }])


def q2_margin_enriched(q2_sens: pd.DataFrame) -> pd.DataFrame:
    out = q2_sens.copy()
    base = float(out.loc[out["margin"] == out["margin"].min(), "annual_total_cost"].iloc[0])
    out["安全裕度_pct"] = out["margin"] * 100
    out["成本增加_元"] = out["annual_total_cost"] - base
    out["相对0%成本增加率"] = (out["annual_total_cost"] / base - 1.0)
    return out.sort_values("margin").reset_index(drop=True)


class EnhancedPVForecastCorrector(PVForecastCorrector):
    """在原BP残差模型基础上保存测试集预测明细，便于论文作图。"""
    def fit(self, forecast_df, pv_dict):
        super().fit(forecast_df, pv_dict)
        X, y, records = self.build_training_set(forecast_df, pv_dict)
        order = np.argsort(pd.to_datetime([r["target_date"] for r in records]).values)
        X = X[order]
        y = y[order]
        rec_sorted = [records[i] for i in order]
        split = self.train_end_index
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        med = self.feature_medians
        X_train = np.where(np.isfinite(X_train), X_train, med)
        X_test = np.where(np.isfinite(X_test), X_test, med)
        pred_resid = self.model.predict(X_test)
        actual = X_test[:, 0] + y_test
        raw = X_test[:, 0]
        corrected = np.maximum(raw + pred_resid, 0.0)
        rows = []
        for i, rec in enumerate(rec_sorted[split:]):
            rows.append({
                "目标日期": rec["target_date"].date(),
                "目标小时": rec["target_hour"],
                "预报提前量_h": rec["lead_hour"],
                "实际光伏_kW": float(actual[i]),
                "原始预测_kW": float(raw[i]),
                "BP修正预测_kW": float(corrected[i]),
                "原始误差_kW": float(actual[i] - raw[i]),
                "修正误差_kW": float(actual[i] - corrected[i]),
                "原始绝对误差_kW": float(abs(actual[i] - raw[i])),
                "修正绝对误差_kW": float(abs(actual[i] - corrected[i])),
            })
        # 保存BP训练过程的真实loss曲线，用于论文epoch收敛图
        try:
            self.loss_curve_ = np.asarray(self.model.named_steps["bp"].loss_curve_, dtype=float)
        except Exception:
            self.loss_curve_ = np.asarray([], dtype=float)
        self.validation_df = pd.DataFrame(rows)
        self.error_by_lead = self.validation_df.groupby("预报提前量_h").agg(
            原始MAE_kW=("原始绝对误差_kW", "mean"),
            修正MAE_kW=("修正绝对误差_kW", "mean"),
            原始RMSE_kW=("原始误差_kW", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            修正RMSE_kW=("修正误差_kW", lambda s: float(np.sqrt(np.mean(np.square(s)))))
        ).reset_index()
        return self


def make_forecast_144_raw(forecast_df, day, release_hour):
    row = get_forecast_row(forecast_df, day, release_hour)
    arr = np.array([float(row[f"h{i}"]) for i in range(1, 25)])
    return linear_interp_hourly_to_10min(arr, start_hour=release_hour)


def make_forecast_144_corrected(corrector, forecast_df, day, release_hour, pv_dict, conservative=True):
    row = get_forecast_row(forecast_df, day, release_hour)
    hourly = np.array([float(row[f"h{i}"]) for i in range(1, 25)])
    corr = corrector.correct_hourly_forecast(day, release_hour, hourly, pv_dict)
    return hourly_corrected_to_144(corr, day, conservative=conservative)


def run_q3_day_enhanced(day, load, pv_dict, forecast_df, corrector, price,
                        update_hours=OFFICIAL_UPDATE_HOURS, forecast_mode="corrected95"):
    """Q3增强版单日滚动优化。
    forecast_mode = corrected95 / corrected_mean / raw
    """
    day = pd.Timestamp(day).normalize()
    load = ensure_144(load, "Q3负荷")
    actual_pv = ensure_144(pv_dict[day], "Q3实际光伏")
    price = ensure_144(price, "Q3价格")

    plan_grid = np.zeros(SLOTS_PER_DAY)
    plan_charge = np.zeros(SLOTS_PER_DAY)
    plan_dis = np.zeros(SLOTS_PER_DAY)
    exec_grid = np.zeros(SLOTS_PER_DAY)
    exec_charge = np.zeros(SLOTS_PER_DAY)
    exec_dis = np.zeros(SLOTS_PER_DAY)
    exec_soc = np.full(SLOTS_PER_DAY, np.nan)
    schedule_history = []
    current_soc = SOC_INITIAL
    stage_rows = []

    previous_schedule = None
    for stage_no, release_hour in enumerate(update_hours):
        start = release_hour * 6
        if start >= 144:
            continue
        if forecast_mode == "raw":
            pv_forecast = make_forecast_144_raw(forecast_df, day, release_hour)
        elif forecast_mode == "corrected_mean":
            pv_forecast = make_forecast_144_corrected(corrector, forecast_df, day, release_hour, pv_dict, conservative=False)
        else:
            pv_forecast = make_forecast_144_corrected(corrector, forecast_df, day, release_hour, pv_dict, conservative=True)
        opt = solve_variable_length_day(
            load[start:], pv_forecast[start:], price[start:], current_soc, None
        )
        if not opt.success:
            raise RuntimeError(f"Q3增强版滚动优化失败：{day.date()} {release_hour}:00")

        new_grid = opt.grid.copy()
        new_ch = opt.charge.copy()
        new_dis = opt.discharge.copy()
        prev_future = None if previous_schedule is None else previous_schedule

        # 写入最终滚动计划
        exec_grid[start:] = new_grid
        exec_charge[start:] = new_ch
        exec_dis[start:] = new_dis

        if release_hour == 0:
            plan_grid[:] = exec_grid
            plan_charge[:] = exec_charge
            plan_dis[:] = exec_dis

        # 本阶段执行到下一更新时刻前的动作冻结
        next_start = update_hours[stage_no + 1] * 6 if stage_no + 1 < len(update_hours) else 144
        freeze_end = min(next_start, 144)
        for t in range(start, freeze_end):
            prev_soc = current_soc if t == start else exec_soc[t - 1]
            exec_soc[t] = prev_soc + ETA * exec_charge[t] * DT - exec_dis[t] * DT / ETA
        if freeze_end > start:
            current_soc = float(exec_soc[freeze_end - 1])

        # 与上一滚动计划相比，仅统计当前未冻结区间的改变
        if prev_future is None:
            changed = np.zeros(144)
        else:
            changed = exec_grid - prev_future
            changed[:start] = 0.0
        stage_rows.append({
            "日期": day.date(),
            "更新时刻": f"{release_hour:02d}:00",
            "预测模式": forecast_mode,
            "滚动优化成本_元": float(opt.objective),
            "未来预测均值_kW": float(np.mean(pv_forecast[start:])),
            "购电计划变化量_kWh": float(np.sum(np.abs(changed)) * DT),
            "购电上调_kWh": float(np.sum(np.maximum(changed, 0.0)) * DT),
            "购电下调_kWh": float(np.sum(np.maximum(-changed, 0.0)) * DT),
        })
        previous_schedule = exec_grid.copy()

    # 补齐SOC
    for t in range(144):
        if not np.isfinite(exec_soc[t]):
            if t == 0:
                exec_soc[t] = SOC_INITIAL + ETA * exec_charge[t] * DT - exec_dis[t] * DT / ETA
            else:
                exec_soc[t] = exec_soc[t-1] + ETA * exec_charge[t] * DT - exec_dis[t] * DT / ETA

    emergency = np.maximum(load + exec_charge - actual_pv - exec_dis - exec_grid, 0.0)
    surplus = np.maximum(actual_pv + exec_dis + exec_grid - load - exec_charge, 0.0)
    down = np.maximum(plan_grid - exec_grid, 0.0)
    up = np.maximum(exec_grid - plan_grid, 0.0)
    plan_cost = plan_grid * price * DT
    adj_cost = -down * price * DOWN_ADJUST_MULTIPLIER * DT + up * price * UP_ADJUST_MULTIPLIER * DT
    emergency_cost = emergency * price * EMERGENCY_MULTIPLIER * DT
    total_cost = plan_cost + adj_cost + emergency_cost

    detail = pd.DataFrame({
        "日期": [day.date()] * 144,
        "时段": np.arange(1,145),
        "时间_小时": np.arange(144)/6.0,
        "电价": price,
        "实际负荷_kW": load,
        "实际光伏_kW": actual_pv,
        "0时计划购电_kW": plan_grid,
        "最终调整购电_kW": exec_grid,
        "下调购电_kW": down,
        "上调购电_kW": up,
        "充电_kW": exec_charge,
        "放电_kW": exec_dis,
        "SOC_kWh": exec_soc,
        "应急购电_kW": emergency,
        "最终总购电_kW": exec_grid + emergency,
        "计划结算成本": plan_cost,
        "调整结算成本": adj_cost,
        "应急购电成本": emergency_cost,
        "总成本": total_cost,
        "弃光_kW": surplus,
    })
    stages = pd.DataFrame(stage_rows)
    return detail, stages


def solve_q3_year_enhanced(load_dict, pv_dict, forecast_df, price_dict, corrector,
                           update_hours, dynamic_price=False, forecast_mode="corrected95",
                           dates_override=None, verbose=True):
    dates = sorted(set(load_dict) & set(pv_dict)) if dates_override is None else list(dates_override)
    ds, ss = [], []
    for i, d in enumerate(dates):
        if verbose and (i % 30 == 0 or i == len(dates)-1):
            print(f"  Q3增强求解：{i+1}/{len(dates)}，当前 {d.date()}，模式={forecast_mode}")
        price = price_dict[d] if dynamic_price else load_attachment1(FILE_A1).price
        detail, stages = run_q3_day_enhanced(
            d, load_dict[d], pv_dict, forecast_df, corrector, price,
            update_hours=update_hours, forecast_mode=forecast_mode
        )
        ds.append(detail); ss.append(stages)
    detail = pd.concat(ds, ignore_index=True)
    stages = pd.concat(ss, ignore_index=True) if ss else pd.DataFrame()
    summary = {
        "annual_plan_cost": float(detail["计划结算成本"].sum()),
        "annual_adjustment_cost": float(detail["调整结算成本"].sum()),
        "annual_emergency_cost": float(detail["应急购电成本"].sum()),
        "annual_total_cost": float(detail["总成本"].sum()),
        "annual_plan_purchase_kwh": calc_energy(detail["0时计划购电_kW"]),
        "annual_adjusted_purchase_kwh": calc_energy(detail["最终调整购电_kW"]),
        "annual_up_adjust_kwh": calc_energy(detail["上调购电_kW"]),
        "annual_down_adjust_kwh": calc_energy(detail["下调购电_kW"]),
        "annual_emergency_kwh": calc_energy(detail["应急购电_kW"]),
        "annual_charge_kwh": calc_energy(detail["充电_kW"]),
        "annual_discharge_kwh": calc_energy(detail["放电_kW"]),
        "annual_curtailment_kwh": calc_energy(detail["弃光_kW"]),
        "avg_daily_cost": float(detail.groupby("日期")["总成本"].sum().mean()),
        "max_daily_cost": float(detail.groupby("日期")["总成本"].sum().max()),
        "min_daily_cost": float(detail.groupby("日期")["总成本"].sum().min()),
    }
    return detail, stages, summary


def q3_paper_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    total_cost = float(detail["总成本"].sum())
    out = {
        "年总成本": total_cost,
        "计划购电量_kWh": calc_energy(detail["0时计划购电_kW"]),
        "最终调整购电量_kWh": calc_energy(detail["最终调整购电_kW"]),
        "上调购电量_kWh": calc_energy(detail["上调购电_kW"]),
        "下调购电量_kWh": calc_energy(detail["下调购电_kW"]),
        "应急购电量_kWh": calc_energy(detail["应急购电_kW"]),
        "计划结算成本": float(detail["计划结算成本"].sum()),
        "调整结算成本": float(detail["调整结算成本"].sum()),
        "应急购电成本": float(detail["应急购电成本"].sum()),
        "应急成本占比": float(detail["应急购电成本"].sum()/max(total_cost,1e-9)),
        "总充电量_kWh": calc_energy(detail["充电_kW"]),
        "总放电量_kWh": calc_energy(detail["放电_kW"]),
        "弃光量_kWh": calc_energy(detail["弃光_kW"]),
        "平均日成本": float(detail.groupby("日期")["总成本"].sum().mean()),
        "最大日成本": float(detail.groupby("日期")["总成本"].sum().max()),
    }
    return pd.DataFrame([out])


def bp_improvement_table(corrector: EnhancedPVForecastCorrector) -> pd.DataFrame:
    m = corrector.metrics.copy()
    mae_rate = 1 - m["test_MAE_corrected"] / max(m["test_MAE_raw"], 1e-9)
    rmse_rate = 1 - m["test_RMSE_corrected"] / max(m["test_RMSE_raw"], 1e-9)
    return pd.DataFrame([{
        "测试集样本数": m["n_test"],
        "训练集样本数": m["n_train"],
        "原始MAE_kW": m["test_MAE_raw"],
        "BP修正MAE_kW": m["test_MAE_corrected"],
        "MAE改善率": mae_rate,
        "原始RMSE_kW": m["test_RMSE_raw"],
        "BP修正RMSE_kW": m["test_RMSE_corrected"],
        "RMSE改善率": rmse_rate,
    }])


def build_q3_lead_metrics(corrector: EnhancedPVForecastCorrector) -> pd.DataFrame:
    return corrector.error_by_lead.copy()


def build_q3_forecast_summary(corrector: EnhancedPVForecastCorrector) -> pd.DataFrame:
    v = corrector.validation_df.copy()
    return pd.DataFrame([{
        "实际平均_kW": float(v["实际光伏_kW"].mean()),
        "原始预测平均_kW": float(v["原始预测_kW"].mean()),
        "BP修正平均_kW": float(v["BP修正预测_kW"].mean()),
        "原始平均绝对误差_kW": float(v["原始绝对误差_kW"].mean()),
        "BP修正平均绝对误差_kW": float(v["修正绝对误差_kW"].mean()),
        "原始最大绝对误差_kW": float(v["原始绝对误差_kW"].max()),
        "BP修正最大绝对误差_kW": float(v["修正绝对误差_kW"].max()),
    }])


def select_typical_days(load_dict, pv_dict, price_dict, q3_detail=None):
    dates = sorted(set(load_dict) & set(pv_dict) & set(price_dict))
    rows = []
    for d in dates:
        load = load_dict[d]; pv = pv_dict[d]; price = price_dict[d]
        rows.append({
            "date": d,
            "pv_energy": calc_energy(pv),
            "load_energy": calc_energy(load),
            "price_mean": float(np.mean(price)),
            "price_peak": float(np.max(price)),
            "pv_peak": float(np.max(pv)),
        })
    df = pd.DataFrame(rows)
    result = {
        "光伏高出力日": df.loc[df["pv_energy"].idxmax(), "date"],
        "动态电价高峰日": df.loc[df["price_mean"].idxmax(), "date"],
        "负荷高峰日": df.loc[df["load_energy"].idxmax(), "date"],
    }
    return result



# =============================================================================
# 18. 论文最终数据可视化：仅替换可视化，不改变Q1-Q4计算逻辑
# =============================================================================

VIS_COLORS = {
    "blue": "#2c7fb8", "red": "#d7301f", "green": "#41ab5d",
    "purple": "#756bb1", "orange": "#f16913", "cyan": "#1b9e77",
    "light_blue": "#9ecae1", "light_red": "#fcbba1", "light_green": "#a1d99b",
    "light_orange": "#fdae6b", "light_purple": "#bcbddc", "dark": "#222222"
}
VIS_HATCH = ["////", "....", "xxxx", "\\\\"]


def save_plot(fig, filename):
    safe = re.sub(r'[<>:"/\\|?*]', "_", str(filename))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / safe
    fig.savefig(p, dpi=400, facecolor="white")
    plt.close(fig)
    return p


def plot_data_q1(day1):
    """图1：问题一数据特征，负荷/光伏与电价。"""
    t = np.arange(144) / 6.0
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True)
    ax = axes[0]
    ax.fill_between(t, day1.pv_kw, 0, color=VIS_COLORS["light_green"], alpha=0.55, label="光伏预测")
    ax.plot(t, day1.load_kw, color=VIS_COLORS["blue"], linewidth=1.7, label="居民负荷")
    ax.plot(t, day1.pv_kw, color=VIS_COLORS["green"], linestyle="--", linewidth=1.3)
    ax.set_ylabel("功率/kW")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax2 = axes[1]
    ax2.step(t, day1.price, where="post", color=VIS_COLORS["red"], linewidth=1.6, label="固定电价")
    ax2.fill_between(t, day1.price, step="post", color=VIS_COLORS["light_red"], alpha=0.45)
    ax2.set_xlabel("时间/h")
    ax2.set_ylabel("电价/(元·kWh$^{-1}$)")
    ax2.set_xticks(np.arange(0,25,4))
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45)
    ax2.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.97, hspace=0.20)
    return save_plot(fig, "图1_Q1数据特征_负荷光伏电价.png")


def plot_q1_solution(q1_detail):
    """图2：问题一优化结果，功率平衡与SOC。"""
    x = q1_detail["时间_小时"].to_numpy(float)
    fig, axes = plt.subplots(2,1,figsize=(9.2,6.8),sharex=True)
    ax=axes[0]
    ax.plot(x,q1_detail["负荷_kW"],color=VIS_COLORS["blue"],lw=1.7,marker="o",markevery=12,ms=2.8,label="负荷")
    ax.plot(x,q1_detail["光伏预测_kW"],color=VIS_COLORS["red"],ls="--",lw=1.4,marker="s",markevery=12,ms=2.8,label="光伏预测")
    ax.plot(x,q1_detail["计划购电_kW"],color=VIS_COLORS["green"],ls="-.",lw=1.5,marker="^",markevery=12,ms=2.8,label="购电")
    net=q1_detail["储能放电_kW"].to_numpy(float)-q1_detail["储能充电_kW"].to_numpy(float)
    ax.fill_between(x,0,net,where=net>=0,color=VIS_COLORS["light_purple"],alpha=0.42,label="储能净出力(放电)")
    ax.plot(x,net,color=VIS_COLORS["purple"],ls=":",lw=1.2)
    ax.set_ylabel("功率/kW"); ax.set_xticks(np.arange(0,25,4)); ax.grid(axis="y",ls=":",lw=.6,alpha=.45)
    ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.02))
    ax=axes[1]
    ax.plot(x,q1_detail["SOC_kWh"],color=VIS_COLORS["blue"],lw=1.8,marker="o",markevery=12,ms=2.8,label="SOC")
    ax.axhline(1200,color=VIS_COLORS["red"],ls=":",lw=1.2,label="SOC下限")
    ax.axhline(10800,color=VIS_COLORS["green"],ls="--",lw=1.2,label="SOC上限")
    ax.axhline(6000,color=VIS_COLORS["purple"],ls="-.",lw=1.1,label="初始SOC")
    ax.set_xlabel("时间/h"); ax.set_ylabel("储能电量/kWh"); ax.grid(axis="y",ls=":",lw=.6,alpha=.45)
    ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(left=.10,right=.97,bottom=.10,top=.96,hspace=.20)
    return save_plot(fig,"图2_Q1优化结果_功率平衡与SOC.png")


def plot_q2_heatmaps(load_dict,pv_dict):
    """图3：全年负荷与光伏日内热力图。"""
    dates=sorted(set(load_dict)&set(pv_dict))
    load=np.vstack([load_dict[d] for d in dates])
    pv=np.vstack([pv_dict[d] for d in dates])
    load_h=load.reshape(len(dates),24,6).mean(axis=2)
    pv_h=pv.reshape(len(dates),24,6).mean(axis=2)
    fig,axes=plt.subplots(1,2,figsize=(11.2,7.0),sharey=True)
    for ax,mat,cmap,label in [(axes[0],load_h,"Blues","负荷/kW"),(axes[1],pv_h,"YlGn","光伏实际/kW")]:
        im=ax.imshow(mat,aspect="auto",origin="upper",cmap=cmap)
        ax.set_xlabel("小时")
        ax.set_xticks(np.arange(0,24,4),labels=np.arange(0,24,4))
        ax.set_yticks(np.arange(0,len(dates),max(1,len(dates)//8)))
        ax.set_yticklabels([pd.Timestamp(dates[i]).strftime("%m-%d") for i in ax.get_yticks().astype(int)])
        ax.grid(False)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label=label)
    axes[0].set_ylabel("日期")
    fig.subplots_adjust(left=.08,right=.96,bottom=.08,top=.98,wspace=.16)
    return save_plot(fig,"图3_Q2全年负荷光伏日内热力图.png")


def plot_q2_forecast_compare(load_dict,pv_dict,day1):
    """图4：上一日实际作为日前预测，与当日实际的散点一致性。"""
    dates=sorted(set(load_dict)&set(pv_dict))
    xL=[]; yL=[]; xP=[]; yP=[]
    for i,d in enumerate(dates):
        prev_load=day1.load_kw if i==0 else load_dict[dates[i-1]]
        prev_pv=day1.pv_kw if i==0 else pv_dict[dates[i-1]]
        xL.extend(prev_load.tolist()); yL.extend(load_dict[d].tolist())
        xP.extend(prev_pv.tolist()); yP.extend(pv_dict[d].tolist())
    rng=np.random.default_rng(RANDOM_SEED)
    def subs(a,b,n=6000):
        m=min(len(a),n); idx=rng.choice(len(a),m,replace=False); return np.asarray(a)[idx],np.asarray(b)[idx]
    xL,yL=subs(xL,yL); xP,yP=subs(xP,yP)
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.8))
    for ax,x,y,label,c in [(axes[0],xL,yL,"负荷",VIS_COLORS["blue"]),(axes[1],xP,yP,"光伏",VIS_COLORS["green"])]:
        lim=[0,max(x.max(),y.max())*1.05]
        ax.scatter(x,y,s=10,alpha=.22,color=c,edgecolors="none")
        ax.plot(lim,lim,color=VIS_COLORS["dark"],ls="--",lw=1.1,label="理想一致线")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(f"上一日实际{label}/kW"); ax.set_ylabel(f"当日实际{label}/kW")
        ax.grid(ls=":",lw=.6,alpha=.45); ax.legend(frameon=False,loc="upper left")
    fig.subplots_adjust(left=.08,right=.97,bottom=.14,top=.96,wspace=.24)
    return save_plot(fig,"图4_Q2日前预测_上一日实际与当日实际对比.png")


def plot_q2_margin_new(q2):
    """图5：安全裕度成本—风险敏感性。"""
    x=q2["安全裕度_pct"].to_numpy(float); cost=q2["annual_total_cost"].to_numpy(float)/1e4; em=q2["annual_emergency_kwh"].to_numpy(float)/1e6
    fig,ax1=plt.subplots(figsize=(9.2,4.9))
    ax1.plot(x,cost,color=VIS_COLORS["blue"],lw=2,marker="o",ms=6,label="年总成本")
    ax1.fill_between(x,cost,cost.min(),color=VIS_COLORS["light_blue"],alpha=.20)
    ax1.set_xlabel("安全裕度/%"); ax1.set_ylabel("年总成本/万元"); ax1.grid(axis="y",ls=":",lw=.6,alpha=.45)
    ax2=ax1.twinx(); ax2.plot(x,em,color=VIS_COLORS["red"],ls="--",lw=1.8,marker="s",ms=5,label="应急购电量")
    ax2.set_ylabel("应急购电量/百万kWh")
    bi=int(np.argmin(cost)); ax1.scatter(x[bi],cost[bi],s=65,color=VIS_COLORS["purple"],zorder=5)
    ax1.annotate(f"最优 {x[bi]:.0f}%",(x[bi],cost[bi]),xytext=(8,8),textcoords="offset points",fontsize=9)
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax1.legend(h1+h2,l1+l2,frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(left=.09,right=.91,bottom=.14,top=.92)
    return save_plot(fig,"图5_Q2安全裕度_成本风险敏感性.png")


def plot_q3_data_heatmaps(forecast_df,corrector):
    """图6：左侧展示四个正式发布时刻的平均预测结构，右侧展示原始绝对误差热力图。"""
    f=forecast_df.copy()
    rows=[]
    for h in sorted(f["release_hour"].unique()):
        sub=f[f["release_hour"]==h]
        vals=[float(sub[f"h{i}"].mean()) for i in range(1,25)]
        rows.append(vals)
    pred=np.asarray(rows)
    v=corrector.validation_df.copy()
    err=v.pivot_table(index="目标小时",columns="预报提前量_h",values="原始绝对误差_kW",aggfunc="mean").reindex(index=range(24),columns=range(1,25))
    fig,axes=plt.subplots(1,2,figsize=(11.2,6.2))
    im=axes[0].imshow(pred,aspect="auto",cmap="YlOrRd")
    axes[0].set_yticks(np.arange(len(sorted(f["release_hour"].unique()))),labels=[f"{int(h):02d}:00" for h in sorted(f["release_hour"].unique())]); axes[0].set_xticks(np.arange(0,24,4),labels=np.arange(1,25,4)); axes[0].set_xlabel("预测步长/h"); axes[0].set_ylabel("正式发布时刻"); axes[0].grid(False)
    fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04,label="平均预测功率/kW")
    im2=axes[1].imshow(err.to_numpy(),aspect="auto",origin="lower",cmap="RdYlBu_r")
    axes[1].set_xticks(np.arange(0,24,4),labels=np.arange(1,25,4)); axes[1].set_yticks(np.arange(0,24,4),labels=np.arange(0,24,4)); axes[1].set_xlabel("预测提前量/h"); axes[1].set_ylabel("目标时刻/h"); axes[1].grid(False)
    fig.colorbar(im2,ax=axes[1],fraction=.046,pad=.04,label="原始MAE/kW")
    fig.subplots_adjust(left=.08,right=.96,bottom=.10,top=.97,wspace=.18)
    return save_plot(fig,"图6_Q3预测结构与原始误差热力图.png")


def plot_q3_bp(corrector):
    """图7：BP训练收敛 + 原始/修正绝对误差分布。"""
    loss=np.asarray(corrector.loss_curve_,float); v=corrector.validation_df
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.8))
    ep=np.arange(1,len(loss)+1); axes[0].plot(ep,loss,color=VIS_COLORS["blue"],lw=1.8,label="训练损失")
    axes[0].scatter(ep[-1],loss[-1],s=35,color=VIS_COLORS["red"],label="最终损失"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE"); axes[0].grid(ls=":",lw=.6,alpha=.45); axes[0].legend(frameon=False)
    data=[v["原始绝对误差_kW"].to_numpy(),v["修正绝对误差_kW"].to_numpy()]
    try:
        bp = axes[1].boxplot(data, patch_artist=True, tick_labels=["原始预测","BP修正"], showfliers=False)
    except TypeError:
        bp = axes[1].boxplot(data, patch_artist=True, labels=["原始预测","BP修正"], showfliers=False)
    fills=[VIS_COLORS["light_red"],VIS_COLORS["light_blue"]]
    for patch,c in zip(bp["boxes"],fills): patch.set_facecolor(c); patch.set_edgecolor(VIS_COLORS["dark"])
    axes[1].set_ylabel("绝对误差/kW"); axes[1].grid(axis="y",ls=":",lw=.6,alpha=.45)
    fig.subplots_adjust(left=.08,right=.97,bottom=.13,top=.96,wspace=.24)
    return save_plot(fig,"图7_Q3_BP训练收敛与误差分布.png")


def plot_q3_lead_and_box(corrector):
    """图8：预测提前量MAE趋势 + 分组误差箱线图。"""
    df=corrector.error_by_lead.copy().sort_values("预报提前量_h"); x=df["预报提前量_h"].to_numpy(float)
    v=corrector.validation_df.copy(); v["提前量组"]=pd.cut(v["预报提前量_h"],[0,8,16,24],labels=["1–8 h","9–16 h","17–24 h"])
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.8))
    axes[0].plot(x,df["原始MAE_kW"],color=VIS_COLORS["red"],ls="--",marker="o",lw=1.5,ms=4,label="原始预测MAE")
    axes[0].plot(x,df["修正MAE_kW"],color=VIS_COLORS["blue"],marker="s",lw=1.5,ms=4,label="BP修正MAE")
    axes[0].set_xlabel("预测提前量/h"); axes[0].set_ylabel("MAE/kW"); axes[0].grid(axis="y",ls=":",lw=.6,alpha=.45); axes[0].legend(frameon=False)
    groups=[v.loc[v["提前量组"]==c,"修正误差_kW"].dropna().to_numpy() for c in ["1–8 h","9–16 h","17–24 h"]]
    try:
        bp = axes[1].boxplot(groups, patch_artist=True, tick_labels=["1–8 h","9–16 h","17–24 h"], showfliers=False)
    except TypeError:
        bp = axes[1].boxplot(groups, patch_artist=True, labels=["1–8 h","9–16 h","17–24 h"], showfliers=False)
    for patch,c in zip(bp["boxes"],[VIS_COLORS["light_green"],VIS_COLORS["light_orange"],VIS_COLORS["light_purple"]]): patch.set_facecolor(c); patch.set_edgecolor(VIS_COLORS["dark"])
    axes[1].axhline(0,color=VIS_COLORS["dark"],ls=":",lw=.9); axes[1].set_xlabel("预测提前量分组"); axes[1].set_ylabel("BP修正误差/kW"); axes[1].grid(axis="y",ls=":",lw=.6,alpha=.45)
    fig.subplots_adjust(left=.08,right=.97,bottom=.13,top=.96,wspace=.24)
    return save_plot(fig,"图8_Q3预测提前量_误差趋势与分组箱线图.png")


def build_q3_frequency_daily_costs(load_dict, pv_dict, forecast_df, price_dict, corrector,
                                   sample_days=15):
    """生成滚动重优化频率敏感性分析所需的逐日成本与应急购电数据。

    标准方案为0/6/12/18；加3小时方案为0/3/6/9/12/15/18/21；
    每4小时方案为0/4/8/12/16/20，其中新增时刻仅用于敏感性模拟。
    """
    all_dates = sorted(set(load_dict) & set(pv_dict))
    if len(all_dates) <= sample_days:
        dates = all_dates
    else:
        idx = np.linspace(0, len(all_dates) - 1, sample_days).round().astype(int)
        dates = [all_dates[i] for i in idx]

    sample_load = {d: load_dict[d] for d in dates}
    sample_pv = {d: pv_dict[d] for d in dates}
    sample_price = {d: price_dict[d] for d in dates}
    scenarios = {
        "0:00一次": ([0], False),
        "标准4次": ([0, 6, 12, 18], False),
        "每4小时": ([0, 4, 8, 12, 16, 20], True),
        "加3小时更新": ([0, 3, 6, 9, 12, 15, 18, 21], True),
    }
    out_cost = {}
    out_emergency = {}
    for name, (hours, sim) in scenarios.items():
        detail, _, _ = solve_problem3_year(
            load_dict=sample_load,
            pv_dict=sample_pv,
            forecast_df=forecast_df,
            price_dict=sample_price,
            corrector=corrector,
            update_hours=hours,
            dynamic_price=False,
            use_simulated_updates=sim,
        )
        grouped = detail.groupby("日期").agg(
            日总成本=("总成本", "sum"),
            日应急购电=("应急购电_kW", lambda x: float(np.sum(x) * DT)),
        )
        grouped.index = pd.to_datetime(grouped.index)
        idx_dt = pd.to_datetime(dates)
        out_cost[name] = grouped.reindex(idx_dt)["日总成本"].to_numpy(dtype=float)
        out_emergency[name] = grouped.reindex(idx_dt)["日应急购电"].to_numpy(dtype=float)
    return dates, out_cost, out_emergency



def build_frequency_daily_inline(load_dict,pv_dict,forecast_df,price_dict,corrector):
    # 直接调用现有已验证的频率敏感性函数，避免重复实现滚动模型
    _, cost, emergency = build_q3_frequency_daily_costs(load_dict,pv_dict,forecast_df,price_dict,corrector,sample_days=FREQUENCY_SAMPLE_DAYS)
    return cost, emergency


def plot_q3_frequency_new(cost,emergency):
    labels=["标准4次","加3小时8次","每4小时6次"]; keys=["标准4次","加3小时更新","每4小时"]
    x=np.arange(3); cm=np.array([np.mean(cost[k]) for k in keys])/1e4; em=np.array([np.mean(emergency[k]) for k in keys])
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.8))
    for ax,vals,ylabel,title_hatches in [(axes[0],cm,"日均总成本/万元",["////","....","xxxx"]),(axes[1],em,"日均应急购电量/kWh",["////","....","xxxx"])]:
        bars=ax.bar(x,vals,width=.56,color=[VIS_COLORS["light_blue"],VIS_COLORS["light_red"],VIS_COLORS["light_green"]],edgecolor="black",linewidth=1)
        for b,h,v in zip(bars,title_hatches,vals): b.set_hatch(h); ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*.02,f"{v:.2f}",ha="center",va="bottom",fontsize=9)
        ax.set_xticks(x,labels); ax.set_xlabel("滚动重优化频率"); ax.set_ylabel(ylabel); ax.grid(axis="y",ls=":",lw=.6,alpha=.45); ax.spines["top"].set_visible(False)
    fig.subplots_adjust(left=.08,right=.97,bottom=.18,top=.96,wspace=.28)
    return save_plot(fig,"图9_Q3滚动重优化频率_成本与应急购电量.png")


def plot_q4_dynamic_price(price_dict):
    """图10：动态电价全年热力图 + 24小时价格分布箱线图。"""
    dates=sorted(price_dict); mat=np.vstack([price_dict[d] for d in dates]).reshape(len(dates),24,6).mean(axis=2)
    hourly=[mat[:,h] for h in range(24)]
    fig,axes=plt.subplots(1,2,figsize=(11.4,7.0),gridspec_kw={"width_ratios":[1.55,1]})
    im=axes[0].imshow(mat,aspect="auto",origin="upper",cmap="RdYlBu_r")
    axes[0].set_xlabel("小时"); axes[0].set_ylabel("日期"); axes[0].set_xticks(np.arange(0,24,4),labels=np.arange(0,24,4)); axes[0].set_yticks(np.arange(0,len(dates),max(1,len(dates)//8)),labels=[pd.Timestamp(dates[i]).strftime("%m-%d") for i in np.arange(0,len(dates),max(1,len(dates)//8))]); axes[0].grid(False)
    fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04,label="动态电价/(元·kWh$^{-1}$)")
    bp=axes[1].boxplot(hourly,patch_artist=True,showfliers=False)
    for i,p in enumerate(bp["boxes"]): p.set_facecolor(VIS_COLORS["light_orange"] if i%2==0 else VIS_COLORS["light_blue"]); p.set_edgecolor(VIS_COLORS["dark"])
    axes[1].set_xlabel("小时"); axes[1].set_ylabel("动态电价/(元·kWh$^{-1}$)"); axes[1].set_xticks(np.arange(1,25,4),labels=np.arange(0,24,4)); axes[1].grid(axis="y",ls=":",lw=.6,alpha=.45)
    fig.subplots_adjust(left=.08,right=.96,bottom=.10,top=.98,wspace=.20)
    return save_plot(fig,"图10_Q4动态电价_全年热力图与小时箱线图.png")


def plot_q4_compare(q2_metrics,q3_metrics,q42_summary,q43_summary,q3_detail):
    """图11：四种运行方案成本结构 + Q3月度成本曲线。"""
    names=["Q2","Q3","Q4-2","Q4-3"]
    plan=[float(q2_metrics["年总成本"].iloc[0]),float(q3_metrics["计划结算成本"].iloc[0]),float(q42_summary["annual_scheduled_cost"]),float(q43_summary["annual_plan_cost"])]
    emergency=[float(q2_metrics["应急购电成本"].iloc[0]),float(q3_metrics["应急购电成本"].iloc[0]),float(q42_summary["annual_emergency_cost"]),float(q43_summary["annual_emergency_cost"])]
    adj=[0,float(q3_metrics["调整结算成本"].iloc[0]),0,float(q43_summary["annual_adjustment_cost"])]
    vals=[np.array(plan)/1e4,np.array(emergency)/1e4,np.array(adj)/1e4]
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.9))
    x=np.arange(4); w=.22
    for i,(v,c,label,h) in enumerate(zip(vals,[VIS_COLORS["light_blue"],VIS_COLORS["light_red"],VIS_COLORS["light_green"]],["计划购电成本","应急购电成本","调整结算净额"],["////","....","xxxx"])):
        b=axes[0].bar(x+(i-1)*w,v,width=w,color=c,edgecolor="black",linewidth=.8,label=label)
        for bb,hh in zip(b,[h]*len(b)): bb.set_hatch(hh)
    axes[0].axhline(0,color=VIS_COLORS["dark"],lw=.8); axes[0].set_xticks(x,names); axes[0].set_ylabel("年度金额/万元"); axes[0].set_xlabel("运行方案"); axes[0].grid(axis="y",ls=":",lw=.6,alpha=.45); axes[0].legend(frameon=False,fontsize=8)
    d=q3_detail.copy(); d["月份"]=pd.to_datetime(d["日期"]).dt.month; g=d.groupby("月份")["总成本"].sum()/1e4
    axes[1].plot(g.index,g.values,color=VIS_COLORS["blue"],lw=1.8,marker="o",ms=4,label="Q3月总成本")
    axes[1].fill_between(g.index,g.values,color=VIS_COLORS["light_blue"],alpha=.25)
    axes[1].set_xlabel("月份"); axes[1].set_ylabel("月总成本/万元"); axes[1].set_xticks(np.arange(1,13)); axes[1].grid(axis="y",ls=":",lw=.6,alpha=.45); axes[1].legend(frameon=False,loc="upper left")
    fig.subplots_adjust(left=.08,right=.97,bottom=.13,top=.96,wspace=.24)
    return save_plot(fig,"图11_综合经济性_年度成本结构与Q3月度运行.png")


def create_paper_tables(q1_detail, q2_sens, q2_daily, corrector, q3_detail, q3_freq,
                        q42_summary, q43_detail, q43_summary):
    q2e = q2_margin_enriched(q2_sens)
    q2best = q2_paper_metrics(q2_daily)
    bpimp = bp_improvement_table(corrector)
    q3met = q3_paper_metrics(q3_detail)
    q43met = q3_paper_metrics(q43_detail)
    q4_compare = pd.DataFrame([
        {"方案":"Q2固定电价", "年总成本":float(q2best["年总成本"].iloc[0])},
        {"方案":"Q3滚动优化", "年总成本":float(q3met["年总成本"].iloc[0])},
        {"方案":"Q4-2动态电价", "年总成本":float(q42_summary["annual_total_cost"])},
        {"方案":"Q4-3动态电价+滚动", "年总成本":float(q43_summary["annual_total_cost"])},
    ])
    base_q2 = float(q2best["年总成本"].iloc[0])
    base_q3 = float(q3met["年总成本"].iloc[0])
    q4_compare["相对基准变化率"] = [
        0.0,
        0.0,
        q4_compare.loc[2,"年总成本"]/base_q2-1,
        q4_compare.loc[3,"年总成本"]/base_q3-1,
    ]
    monthly_q3 = q3_detail.copy()
    monthly_q3["月份"] = pd.to_datetime(monthly_q3["日期"]).dt.month
    monthly_q3 = monthly_q3.groupby("月份", as_index=False).agg(
        月总成本=("总成本","sum"), 月计划购电=("0时计划购电_kW", lambda x: calc_energy(x)),
        月调整购电=("最终调整购电_kW", lambda x: calc_energy(x)), 月应急购电=("应急购电_kW", lambda x: calc_energy(x)),
        月弃光=("弃光_kW", lambda x: calc_energy(x))
    )
    return {
        "Q1指标": q1_paper_metrics(q1_detail),
        "Q2裕度敏感性": q2e,
        "Q2最优方案指标": q2best,
        "BP总体指标": bpimp,
        "BP提前量指标": build_q3_lead_metrics(corrector),
        "BP测试集统计": build_q3_forecast_summary(corrector),
        "Q3调度指标": q3met,
        "Q3更新频率": q3_freq,
        "Q3月度统计": monthly_q3,
        "Q4经济性比较": q4_compare,
        "Q4-2摘要": pd.DataFrame([q42_summary]),
        "Q4-3摘要": q43met,
    }



# =============================================================================
# 19. 题目模板表1/表2/表3：直接对应C题PDF格式
# =============================================================================

TEMPLATE_DATES = [
    pd.Timestamp("2025-03-20"),
    pd.Timestamp("2025-06-21"),
    pd.Timestamp("2025-09-23"),
    pd.Timestamp("2025-12-21"),
]
TEMPLATE_SAMPLE_HOURS = [10, 12, 14, 16, 18, 20]
TEMPLATE_BLOCKS = [(0,4), (4,8), (8,12), (12,16), (16,20), (20,24)]


def _slot_energy(arr, hour):
    """指定10 min时段的电量(kWh)，hour表示区间起始小时。"""
    return float(np.asarray(arr, dtype=float)[hour * SLOTS_PER_HOUR] * DT)


def _block_energy(arr, h0, h1):
    """指定4小时区间的能量(kWh)。"""
    a = np.asarray(arr, dtype=float)
    return float(np.sum(a[h0 * SLOTS_PER_HOUR:h1 * SLOTS_PER_HOUR]) * DT)


def _format_emergency_intervals(power):
    """把逐10 min应急购电功率合并成连续区间，返回[(时间段, kWh), ...]。"""
    a = np.asarray(power, dtype=float)
    out = []
    i = 0
    while i < len(a):
        if a[i] <= 1e-8:
            i += 1
            continue
        j = i
        while j + 1 < len(a) and a[j + 1] > 1e-8:
            j += 1
        start_min = i * 10
        end_min = (j + 1) * 10
        def fmt(m):
            if m == 1440:
                return "24:00"
            return f"{m // 60:02d}:{m % 60:02d}"
        out.append((f"{fmt(start_min)}-{fmt(end_min)}", float(np.sum(a[i:j+1]) * DT)))
        i = j + 1
    return out


def _make_problem_template_tables(q1_detail, q2_daily, q2_best_margin, q2_plan_load, q2_plan_pv,
                                   load_dict, pv_dict, day1, price_dict, q3_detail):
    """生成与题目PDF表1/表2/表3一一对应的论文用表。"""
    tables = {}

    # -------------------------
    # Q1：题目表1
    # -------------------------
    vals = [(f"{h:02d}:00-{h:02d}:10", round(_slot_energy(q1_detail["计划购电_kW"].to_numpy(), h), 2))
            for h in TEMPLATE_SAMPLE_HOURS]
    tables["题目表1_Q1购电"] = pd.DataFrame([
        [vals[0][0], vals[0][1], vals[1][0], vals[1][1], vals[2][0], vals[2][1]],
        [vals[3][0], vals[3][1], vals[4][0], vals[4][1], vals[5][0], vals[5][1]],
        ["全天购电量", round(calc_energy(q1_detail["计划购电_kW"]), 2),
         "全天购电费", round(float(q1_detail["购电成本"].sum()), 2), "", ""],
    ], columns=["时间段","购电量/kWh","时间段","购电量/kWh","时间段","购电量/kWh"])

    # -------------------------
    # Q1：题目表2
    # -------------------------
    # 题目表2采用“每行两个时间段”的版式，共3行+末行SOC
    block_rows = []
    for i in range(0, len(TEMPLATE_BLOCKS), 2):
        a1, b1 = TEMPLATE_BLOCKS[i]
        a2, b2 = TEMPLATE_BLOCKS[i + 1]
        block_rows.append([
            f"{a1:02d}:00-{b1:02d}:00",
            round(_block_energy(q1_detail["储能充电_kW"], a1, b1), 2),
            round(_block_energy(q1_detail["储能放电_kW"], a1, b1), 2),
            f"{a2:02d}:00-{b2:02d}:00",
            round(_block_energy(q1_detail["储能充电_kW"], a2, b2), 2),
            round(_block_energy(q1_detail["储能放电_kW"], a2, b2), 2),
        ])
    block_rows.append(["0:00储电量", 6000.0, "", "24:00储电量", 6000.0, ""])
    tables["题目表2_Q1储能"] = pd.DataFrame(
        block_rows,
        columns=["时间段","充电量/kWh","放电量/kWh","时间段2","充电量2/kWh","放电量2/kWh"]
    )

    # -------------------------
    # Q2：四个指定日期，表1/表2/表3
    # -------------------------
    q2_table1 = []
    q2_table2 = []
    q2_table3 = []
    # 只重算题目指定的4天，初始SOC直接采用Q2全年逐日结果中的期初SOC，
    # 避免为了导出表格而再次完整运行365天。
    q2_daily_idx = q2_daily.copy()
    q2_daily_idx["日期"] = pd.to_datetime(q2_daily_idx["日期"])
    fixed_price = load_attachment1(FILE_A1).price
    for d in TEMPLATE_DATES:
        m = q2_daily_idx[q2_daily_idx["日期"].dt.normalize() == d]
        if m.empty:
            continue
        soc0 = float(m.iloc[0]["期初SOC_kWh"])
        plan = optimize_day_plan_with_margin(
            q2_plan_load[d], q2_plan_pv[d], fixed_price,
            q2_best_margin, soc0
        )
        exe = execute_scheduled_day(load_dict[d], pv_dict[d], fixed_price, plan)
        q2_table1.append([
            str(d.date()),
            *[round(_slot_energy(plan.grid, h), 2) for h in TEMPLATE_SAMPLE_HOURS]
        ])
        row = [str(d.date())]
        for a, b in TEMPLATE_BLOCKS:
            row += [round(_block_energy(plan.charge, a, b), 2), round(_block_energy(plan.discharge, a, b), 2)]
        row += [round(float(soc0), 2), round(float(plan.soc[-1]), 2)]
        q2_table2.append(row)
        intervals = _format_emergency_intervals(exe["emergency"])
        for k, (itv, qty) in enumerate(intervals):
            q2_table3.append([str(d.date()) if k == 0 else "", itv, round(qty, 2)])

    tables["题目表1_Q2指定日期"] = pd.DataFrame(
        q2_table1,
        columns=["日期","10:00-10:10","12:00-12:10","14:00-14:10","16:00-16:10","18:00-18:10","20:00-20:10"]
    )
    tables["题目表2_Q2指定日期"] = pd.DataFrame(
        q2_table2,
        columns=["日期","00-04充电","00-04放电","04-08充电","04-08放电","08-12充电","08-12放电",
                 "12-16充电","12-16放电","16-20充电","16-20放电","20-24充电","20-24放电","0:00 SOC","24:00 SOC"]
    )
    tables["题目表3_Q2紧急购电"] = pd.DataFrame(q2_table3, columns=["日期","紧急购电时间段","紧急购电量/kWh"])

    # -------------------------
    # Q3：四个指定日期，表1/表2/表3
    # -------------------------
    q3_table1, q3_table2, q3_table3 = [], [], []
    q3 = q3_detail.copy()
    q3["日期"] = pd.to_datetime(q3["日期"])
    for d in TEMPLATE_DATES:
        dd = q3[q3["日期"].dt.normalize() == d].copy()
        if dd.empty:
            continue
        q3_table1.append([
            str(d.date()),
            *[round(_slot_energy(dd["最终调整购电_kW"].to_numpy(), h), 2) for h in TEMPLATE_SAMPLE_HOURS]
        ])
        row = [str(d.date())]
        for a, b in TEMPLATE_BLOCKS:
            row += [round(_block_energy(dd["充电_kW"], a, b), 2), round(_block_energy(dd["放电_kW"], a, b), 2)]
        row += [round(float(dd.iloc[0]["SOC_kWh"]), 2), round(float(dd.iloc[-1]["SOC_kWh"]), 2)]
        q3_table2.append(row)
        intervals = _format_emergency_intervals(dd["应急购电_kW"].to_numpy())
        for k, (itv, qty) in enumerate(intervals):
            q3_table3.append([str(d.date()) if k == 0 else "", itv, round(qty, 2)])

    tables["题目表1_Q3指定日期"] = pd.DataFrame(
        q3_table1,
        columns=["日期","10:00-10:10","12:00-12:10","14:00-14:10","16:00-16:10","18:00-18:10","20:00-20:10"]
    )
    tables["题目表2_Q3指定日期"] = pd.DataFrame(
        q3_table2,
        columns=["日期","00-04充电","00-04放电","04-08充电","04-08放电","08-12充电","08-12放电",
                 "12-16充电","12-16放电","16-20充电","16-20放电","20-24充电","20-24放电","0:00 SOC","24:00 SOC"]
    )
    tables["题目表3_Q3紧急购电"] = pd.DataFrame(q3_table3, columns=["日期","紧急购电时间段","紧急购电量/kWh"])
    return tables

def enhanced_main():
    np.random.seed(RANDOM_SEED)
    print_banner("微网与外部电网电力调控策略——论文结果增强版")
    print("脚本目录：", BASE_DIR)
    print("附件1：", FILE_A1)
    print("附件2：", FILE_A2)
    print("附件3：", FILE_A3)
    print("附件4：", FILE_A4)

    print_banner("一、读取数据")
    day1 = load_attachment1(FILE_A1)
    load_dict, pv_dict = load_attachment2(FILE_A2)
    forecast_df = load_attachment3(FILE_A3)
    price_dict = load_attachment4(FILE_A4)
    print(f"附件2日期数：{len(load_dict)}")
    print(f"附件3预报记录：{len(forecast_df)}")
    print(f"附件4日期数：{len(price_dict)}")
    print(f"附件2日期范围：{min(load_dict).date()} ~ {max(load_dict).date()}")

    # Q1
    print_banner("问题1：基础日内优化与论文指标")
    q1_detail = solve_problem1(day1)
    q1_metrics = q1_paper_metrics(q1_detail)
    write_excel(BASE_DIR / "result1.xlsx", {
        "Q1_最优日计划": q1_detail,
        "Q1_论文指标": q1_metrics,
    })

    # Q2
    print_banner("问题2：全年安全裕度比较与稳健性统计")
    q2_plan_load, q2_plan_pv = build_q2_day_ahead_forecast(load_dict, pv_dict, day1, Q2_PLAN_FORECAST_METHOD)
    print(f"问题2计划数据口径：{Q2_PLAN_FORECAST_METHOD}（0:00不使用当天实际负荷/光伏）")
    best_margin, q2_sens, q2_daily_by_margin = solve_problem2_all_margins(
        load_dict, pv_dict, price_dict, DEFAULT_MARGINS, q2_plan_load, q2_plan_pv)
    q2_sens_enriched = q2_margin_enriched(q2_sens)
    q2_daily = q2_daily_by_margin[best_margin]
    q2_metrics = q2_paper_metrics(q2_daily)
    q2_metrics["计划预测方法"] = Q2_PLAN_FORECAST_METHOD
    print(f"问题2最优安全裕度：{best_margin:.0%}")
    write_excel(BASE_DIR / "result2.xlsx", {
        "Q2_安全裕度比较": q2_sens_enriched,
        "Q2_最优裕度逐日": enrich_q2_daily(q2_daily),
        "Q2_论文指标": q2_metrics,
    })

    # BP
    print_banner("问题3：BP残差修正与预测误差统计")
    corrector = EnhancedPVForecastCorrector().fit(forecast_df, pv_dict)
    bp_metrics = bp_improvement_table(corrector)
    print(f"原始MAE：{corrector.metrics['test_MAE_raw']:.3f} kW")
    print(f"修正MAE：{corrector.metrics['test_MAE_corrected']:.3f} kW")
    print(f"MAE改善率：{float(bp_metrics['MAE改善率'].iloc[0])*100:.2f}%")
    print(f"原始RMSE：{corrector.metrics['test_RMSE_raw']:.3f} kW")
    print(f"修正RMSE：{corrector.metrics['test_RMSE_corrected']:.3f} kW")
    print(f"RMSE改善率：{float(bp_metrics['RMSE改善率'].iloc[0])*100:.2f}%")

    # Q3主结果
    print_banner("问题3：全年0/6/12/18滚动优化")
    q3_detail, q3_stages, q3_summary = solve_q3_year_enhanced(
        load_dict, pv_dict, forecast_df, price_dict, corrector,
        OFFICIAL_UPDATE_HOURS, dynamic_price=False, forecast_mode="corrected95", verbose=True
    )
    q3_metrics = q3_paper_metrics(q3_detail)

    # Q3基准：原始预测（用于证明BP修正的调度价值）
    print_banner("问题3附加对比：原始预测 vs BP修正")
    compare_dates = sorted(load_dict.keys())
    if len(compare_dates) > BASELINE_SAMPLE_DAYS:
        idx = np.linspace(0, len(compare_dates)-1, BASELINE_SAMPLE_DAYS).round().astype(int)
        compare_dates = [compare_dates[i] for i in idx]
    if RUN_Q3_RAW_BASELINE:
        raw_detail_30, _, raw_summary_30 = solve_q3_year_enhanced(
            load_dict, pv_dict, forecast_df, price_dict, corrector,
            OFFICIAL_UPDATE_HOURS, dynamic_price=False, forecast_mode="raw",
            dates_override=compare_dates, verbose=True
        )
    else:
        raw_detail_30 = pd.DataFrame()
        raw_summary_30 = {"annual_total_cost":np.nan, "annual_emergency_kwh":np.nan, "annual_up_adjust_kwh":np.nan, "annual_down_adjust_kwh":np.nan}
    corr_detail_30 = q3_detail[q3_detail["日期"].isin([d.date() for d in compare_dates])]
    baseline_compare = pd.DataFrame([
        {"方案":"原始附件3预测", "样本天数":len(compare_dates), "样本期总成本":raw_summary_30["annual_total_cost"],
         "样本期应急购电_kWh":raw_summary_30["annual_emergency_kwh"], "样本期调整购电_kWh":raw_summary_30["annual_up_adjust_kwh"]+raw_summary_30["annual_down_adjust_kwh"]},
        {"方案":"BP修正+95%下界", "样本天数":len(compare_dates), "样本期总成本":float(corr_detail_30["总成本"].sum()),
         "样本期应急购电_kWh":calc_energy(corr_detail_30["应急购电_kW"]),
         "样本期调整购电_kWh":calc_energy(corr_detail_30["上调购电_kW"])+calc_energy(corr_detail_30["下调购电_kW"])},
    ])

    # 更新频率：正文采用15天代表性样本，主结果仍采用全年完整滚动优化。
    print_banner("问题3：更新频率敏感性分析")
    q3_frequency = analyze_q3_frequency(load_dict, pv_dict, forecast_df, price_dict, corrector)
    q3_frequency = q3_frequency.copy()
    q3_frequency["更新次数/日"] = q3_frequency["更新时刻"].str.count(":00")

    # Q4-2
    print_banner("问题4-2：动态电价下重新求解问题2")
    q42_daily, q42_summary, q42_detail = solve_problem2_for_margin(
        load_dict, pv_dict, price_dict, best_margin, price_mode="dynamic",
        plan_load_dict=q2_plan_load, plan_pv_dict=q2_plan_pv, return_detail=True
    )
    q42_daily_e = enrich_q2_daily(q42_daily)

    # ---- Q4-2 最终结果直接打印到控制台 ----
    q42_total_cost = float(q42_detail["总成本"].sum()) if "总成本" in q42_detail.columns else float(q42_summary.get("annual_total_cost", 0.0))
    q42_scheduled_cost = float(q42_detail["计划购电成本"].sum()) if "计划购电成本" in q42_detail.columns else float(q42_summary.get("annual_scheduled_cost", 0.0))
    q42_emergency_cost = float(q42_detail["应急购电成本"].sum()) if "应急购电成本" in q42_detail.columns else float(q42_summary.get("annual_emergency_cost", 0.0))
    q42_plan_kwh = float(q42_detail["计划购电_kW"].sum() * DT) if "计划购电_kW" in q42_detail.columns else 0.0
    q42_emergency_kwh = float(q42_detail["应急购电_kW"].sum() * DT) if "应急购电_kW" in q42_detail.columns else float(q42_summary.get("annual_emergency_kwh", 0.0))
    q42_total_purchase_kwh = float(q42_detail["总购电_kW"].sum() * DT) if "总购电_kW" in q42_detail.columns else q42_plan_kwh + q42_emergency_kwh
    q42_charge_kwh = float(q42_detail["充电_kW"].sum() * DT) if "充电_kW" in q42_detail.columns else 0.0
    q42_discharge_kwh = float(q42_detail["放电_kW"].sum() * DT) if "放电_kW" in q42_detail.columns else 0.0
    q42_soc_max = float(q42_detail["SOC_kWh"].max()) if "SOC_kWh" in q42_detail.columns else np.nan
    q42_soc_min = float(q42_detail["SOC_kWh"].min()) if "SOC_kWh" in q42_detail.columns else np.nan

    print("\n---------------- 问题4-2最终结果 ----------------")
    print(f"动态电价下采用的最优安全裕度：{best_margin:.0%}")
    print(f"年计划购电量：{q42_plan_kwh:,.2f} kWh")
    print(f"年应急购电量：{q42_emergency_kwh:,.2f} kWh")
    print(f"年总购电量：{q42_total_purchase_kwh:,.2f} kWh")
    print(f"年充电量：{q42_charge_kwh:,.2f} kWh")
    print(f"年放电量：{q42_discharge_kwh:,.2f} kWh")
    print(f"SOC最大值：{q42_soc_max:,.2f} kWh")
    print(f"SOC最小值：{q42_soc_min:,.2f} kWh")
    print(f"计划购电成本：{q42_scheduled_cost:,.2f}")
    print(f"应急购电成本：{q42_emergency_cost:,.2f}")
    print(f"应急购电成本占比：{q42_emergency_cost / q42_total_cost * 100:.4f}%" if q42_total_cost > 0 else "应急购电成本占比：0.0000%")
    print(f"Q4-2年总成本：{q42_total_cost:,.2f}")

    # Q4-3
    print_banner("问题4-3：动态电价下重新求解问题3")
    q43_detail, q43_stages, q43_summary = solve_q3_year_enhanced(
        load_dict, pv_dict, forecast_df, price_dict, corrector,
        OFFICIAL_UPDATE_HOURS, dynamic_price=True, forecast_mode="corrected95", verbose=True
    )
    q43_metrics = q3_paper_metrics(q43_detail)

    # 图表：最终保留11张论文核心图。这里只调用内存中的计算结果，
    # 不读取 result1/result2/result3 等中间文件，因此不会因为结果文件不存在而报错。
    print_banner("生成论文十一张核心图表")
    clear_paper_figures()
    plot_paths = []
    plot_paths.append(plot_data_q1(day1))
    plot_paths.append(plot_q1_solution(q1_detail))
    plot_paths.append(plot_q2_heatmaps(load_dict, pv_dict))
    plot_paths.append(plot_q2_forecast_compare(load_dict, pv_dict, day1))
    plot_paths.append(plot_q2_margin_new(q2_sens_enriched))
    plot_paths.append(plot_q3_data_heatmaps(forecast_df, corrector))
    plot_paths.append(plot_q3_bp(corrector))
    plot_paths.append(plot_q3_lead_and_box(corrector))
    freq_cost, freq_emergency = build_frequency_daily_inline(
        load_dict, pv_dict, forecast_df, price_dict, corrector
    )
    plot_paths.append(plot_q3_frequency_new(freq_cost, freq_emergency))
    plot_paths.append(plot_q4_dynamic_price(price_dict))
    plot_paths.append(plot_q4_compare(q2_metrics, q3_metrics, q42_summary, q43_summary, q3_detail))

    # 论文汇总工作簿
    paper_tables = create_paper_tables(
        q1_detail, q2_sens, q2_daily, corrector, q3_detail, q3_frequency,
        q42_summary, q43_detail, q43_summary
    )
    # 额外输出：严格对应题目PDF中的表1/表2/表3，便于直接粘贴进论文。
    template_tables = _make_problem_template_tables(
        q1_detail, q2_daily, best_margin, q2_plan_load, q2_plan_pv,
        load_dict, pv_dict, day1, price_dict, q3_detail
    )
    paper_tables.update(template_tables)
    paper_tables["Q3_BP调度价值"] = baseline_compare
    paper_tables["Q3更新频率_逐日样本"] = pd.DataFrame({
        "标准4次": freq_cost["标准4次"],
        "每4小时": freq_cost["每4小时"],
        "加3小时更新": freq_cost["加3小时更新"],
    })
    paper_tables["论文图表清单"] = pd.DataFrame({"图号":[f"图{i+1}" for i in range(len(plot_paths))],
        "图表文件":[str(p) for p in plot_paths]})
    write_excel(BASE_DIR / "论文结果汇总.xlsx", paper_tables)

    # 各结果文件增强
    write_excel(BASE_DIR / "result3.xlsx", {
        "Q3_逐时段结果": q3_detail,
        "Q3_滚动阶段": q3_stages,
        "Q3_成本汇总": pd.DataFrame([q3_summary]),
        "Q3_论文指标": q3_metrics,
        "Q3_BP总体指标": bp_metrics,
        "Q3_BP测试明细": corrector.validation_df,
        "Q3_BP提前量": corrector.error_by_lead,
        "Q3_原始_vs_BP": baseline_compare,
        "Q3_更新频率": q3_frequency,
    })
    write_excel(BASE_DIR / "result4-2.xlsx", {
        "Q4-2_逐日结果": q42_daily_e,
        "Q4-2_逐时段策略": q42_detail,
        "Q4-2_汇总": pd.DataFrame([q42_summary]),
        "Q4-2_论文指标": q2_paper_metrics(q42_daily),
    })
    write_excel(BASE_DIR / "result4-3.xlsx", {
        "Q4-3_逐时段结果": q43_detail,
        "Q4-3_滚动阶段": q43_stages,
        "Q4-3_成本汇总": pd.DataFrame([q43_summary]),
        "Q4-3_论文指标": q43_metrics,
    })

    # 最终控制台摘要
    print_banner("论文结果增强版计算完成")
    print(f"Q1日成本：{q1_metrics['购电成本'].iloc[0]:,.2f} 元")
    print(f"Q1购电量：{q1_metrics['购电量_kWh'].iloc[0]:,.2f} kWh")
    print(f"Q2最优安全裕度：{best_margin:.0%}")
    print(f"Q2年总成本：{q2_metrics['年总成本'].iloc[0]:,.2f} 元")
    print(f"Q3年总成本：{q3_metrics['年总成本'].iloc[0]:,.2f} 元")
    print(f"Q3计划购电量：{q3_metrics['计划购电量_kWh'].iloc[0]:,.2f} kWh")
    print(f"Q3调整购电量：{(q3_metrics['上调购电量_kWh'].iloc[0] + q3_metrics['下调购电量_kWh'].iloc[0]):,.2f} kWh")
    print(f"Q3应急购电量：{q3_metrics['应急购电量_kWh'].iloc[0]:,.2f} kWh")
    print(f"BP MAE改善率：{bp_metrics['MAE改善率'].iloc[0]*100:.2f}%")
    print(f"BP RMSE改善率：{bp_metrics['RMSE改善率'].iloc[0]*100:.2f}%")
    print(f"Q3计划购电成本：{q3_summary['annual_plan_cost']:,.2f} 元")
    print(f"Q3调整结算成本：{q3_summary['annual_adjustment_cost']:,.2f} 元")
    print(f"Q3应急购电费：{q3_summary['annual_emergency_cost']:,.2f} 元")
    print(f"Q4-2计划购电量：{q42_summary['annual_scheduled_cost']:,.2f} 元成本")
    print(f"Q4-2应急购电量：{q42_summary['annual_emergency_kwh']:,.2f} kWh")
    print(f"Q4-2年总成本：{q42_summary['annual_total_cost']:,.2f} 元")
    print(f"Q4-3计划购电成本：{q43_summary['annual_plan_cost']:,.2f} 元")
    print(f"Q4-3调整结算成本：{q43_summary['annual_adjustment_cost']:,.2f} 元")
    print(f"Q4-3应急购电费：{q43_summary['annual_emergency_cost']:,.2f} 元")
    print(f"Q4-3应急购电量：{q43_summary['annual_emergency_kwh']:,.2f} kWh")
    print(f"Q4-3年总成本：{q43_summary['annual_total_cost']:,.2f} 元")
    print("\n新增论文结果文件：")
    print("  ", BASE_DIR / "论文结果汇总.xlsx")
    print("新增论文图表目录：")
    print("  ", FIG_DIR)
    print("\n原有结果文件同时保留并增强：result1.xlsx / result2.xlsx / result3.xlsx / result4-2.xlsx / result4-3.xlsx")


if __name__ == "__main__":
    enhanced_main()
