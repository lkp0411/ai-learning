"""make_missing.py —— 在 gdp.csv 里人为造一批缺失值，用于练习缺失值处理。

运行一次即可：
    python make_missing.py

会做三件事：
1. 备份原始文件到 gdp_原始备份.csv（已存在则不覆盖）
2. 在 5 个数值列里随机挖掉约 4% 的格子（builtup_area 额外多挖，模拟统计口径缺失）
3. 另外手动写几处「假缺失」标记：N/A 文本、一条横杠 -，用来练习 na_values

随机种子固定为 42，所以每次运行结果一样。
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "gdp.csv"
BAK = HERE / "gdp_原始备份.csv"

rng = np.random.default_rng(42)

df = pd.read_csv(SRC)
n = len(df)

num_cols = ["city_area", "builtup_area", "reg_pop", "resident_pop", "gdp"]

# ---- 1) 随机挖空：每个数值列约 4% ----
for col in num_cols:
    idx = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[idx, col] = np.nan

# ---- 2) builtup_area 额外再挖 10%（模拟"建成区面积"报不全）----
idx = rng.choice(n, size=int(n * 0.10), replace=False)
df.loc[idx, "builtup_area"] = np.nan

# ---- 3) 手动塞几处「看起来像缺失、其实不是 NaN」的标记 ----
# pandas 3.x 不允许往 float64 列里塞字符串，先把这两列转成 object（文本可容纳）
# N/A：pandas 默认就认成 NaN
df["gdp"] = df["gdp"].astype("object")
df.loc[df.index[:3], "gdp"] = "N/A"
# 横杠：pandas 默认【不】认成 NaN，会把这列变成文本类型
df["builtup_area"] = df["builtup_area"].astype("object")
df.loc[df.index[3:5], "builtup_area"] = "-"

df.to_csv(SRC, index=False, float_format="%g")

# ---- 汇报 ----
check = pd.read_csv(SRC)
print("文件已写入:", SRC)
print("形状:", check.shape)
print()
print("各列缺失情况：")
summary = pd.DataFrame(
    {
        "缺失数": check.isna().sum(),
        "缺失率": (check.isna().mean() * 100).round(1).astype(str) + "%",
        "类型": check.dtypes.astype(str),
    }
)
print(summary.to_string())
