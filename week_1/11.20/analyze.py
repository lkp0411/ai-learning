"""analyze.py — 读 gdp.csv，看一眼数据长什么样。

读取方式用了 __file__：不管你在哪个目录运行这个脚本，
它都会去"脚本自己所在的文件夹"找 gdp.csv。
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 脚本所在目录（绝对路径），拼上文件名 —— 这样就不怕"当前工作目录"变来变去
CSV_PATH = Path(__file__).with_name("gdp.csv")

df = pd.read_csv(CSV_PATH)
print(df.shape)
print(df.columns)
print(df.head())
print(df.dtypes)

print(df.isna().sum())  # 每列缺失值数量  isna会把 True/False 转换为 1/0，然后 sum() 就是数量了

df = df.replace(["-", "暂无", "N/A", "null"], np.nan)  # 先把这些字符串替换为 np.nan，方便后续处理

num_cols = ["city_area","builtup_area","reg_pop","resident_pop","gdp"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col],errors="coerce")  # 转为数值型，无法转换的变为 NaN

# ---------- 填充前先留一份"未填充"的副本 ----------
# 后面算基准值、做对比都要用它。千万别在已经填过的列上再算中位数：
# 那样基准会被自己填进去的值带偏（填过一次的列中位数 ≠ 原始中位数）。
raw = df.copy()

# 省内城市数量。上海/北京/天津 这种"直辖市"，province 和 city 同名，
# 组里只有它自己 1 行 —— 中位数就是它自己，一行缺失就等于整组无数可用。
prov_size = raw.groupby("province")["city"].transform("size")
enough = prov_size >= 3   # 省内至少 3 个城市，中位数才有意义

df["gdp"] = df["gdp"].fillna(
    df.groupby("province")["gdp"].transform("median").where(enough) #.where(enough) 的作用：把“城市数 < 3 的省”的中位数全部抹成 NaN，不让它们参与填充。
)
""" 
用省份中位数填充 GDP 缺失值,聚合了省份之中的数据GDP 的省际差异巨大：上海 2 万多亿，小城市几百亿，用全国中位数填上海会严重低估。所以必须按省分组；

面积、人口的省际差异相对小：城市面积、人口虽然也有差异，但不像 GDP 差几十倍，全局中位数填进去的偏差可以接受；

简单优先：能用一个全局值解决，就不引入 groupby 的复杂度。

⚠️ 但有一类省必须排除：上海、北京、天津这种只有 1 个城市的省级单位。
它们的组内中位数 = 它自己，而它自己恰好缺失 → 算出来还是 NaN。
.where(enough) 的作用就是：城市数 < 3 的省，这里直接给 NaN，
让这些城市保持缺失，而不是被下面的全局中位数"猜"一个值填进去。
"""
for col in ["city_area", "reg_pop", "resident_pop"]:
    df[col] = df[col].fillna(raw[col].median())   # 基准用 raw，不用已被填过的 df

"""
GDP 这里刻意**不再用全局中位数兜底**。

原来那句 df["gdp"].fillna(df["gdp"].median()) 会把上海填成 1247 亿
（真实 24965 亿，差 20 倍），因为它只知道"全国中位数"，
不知道"这是全国第一的城市"。

单城市省的 GDP 缺失，靠这份数据本身是推不出来的（没有同省兄弟可参考）。
诚实做法是留 NaN，交给上层决定：要么剔除，要么用别的方法估。
"""
print("填充后仍缺 GDP 的城市：")
print(df[df["gdp"].isna()][["city", "province", "resident_pop"]].to_string(index=False)) #DataFrame 的方法，把整张表转成一个完整字符串。

# builtup_area 缺失率 13.6%，而且是"整块缺失"，硬填会造出一大批假数据。
# 不填，只加一个标记列，让后面分析时能一眼看出这行是缺的。
df["builtup_area_missing"] = df["builtup_area"].isna()

print(df["builtup_area"].unique()[:30])  # unique()：返回这列所有不重复的值，看看到底混了什么；打印前 30 个不同的值，看看数据长什么样

# ---------- 复查：填完必须回头看一眼 ----------
print("=" * 50)
print("填充后各列缺失数：")
print(df[num_cols].isna().sum().to_string())

print()
print("标准差：原始 -> 填充后（填充会压缩方差，这是代价）")
for col in ["city_area", "reg_pop", "resident_pop", "gdp"]:
    print(f"  {col:<14}{raw[col].std():>10.1f} -> {df[col].std():>10.1f}") #字符串前加 f，{} 里可以写表达式，运行时求值填进去。
#col 是列名字符串。"city_area" 9 个字符，左对齐补到 14 宽;{raw[col].std():>10.1f}：右对齐 + 1 位小数;.std()：算标准差
print()
print("GDP × 常住人口 相关系数：%.4f -> %.4f"
      % (raw[["gdp", "resident_pop"]].corr().iloc[0, 1],
         df[["gdp", "resident_pop"]].corr().iloc[0, 1]))

# ---------- 排行榜：不该填充的场景 ----------
# 排序、排名、看"谁最高"这类分析，填充是纯污染：一个假值可能直接改变名次。
# 正确做法是剔除缺失行 —— 332 行去掉 15 行还剩 317 行，完全够用。
rank = raw.dropna(subset=["gdp"])
print()
print(f"排行榜（不填充，直接剔除）：保留 {len(rank)} / {len(raw)} 行，GDP 前十：")
print(rank.nlargest(10, "gdp")[["city", "province", "gdp"]].to_string(index=False))

# ============================================================
# 分组聚合：按省份算 GDP 总和、人口总和、城市数
# ============================================================

# 注意：这里用 df（已填充）还是 raw（未填充）？
# - 算“省份 GDP 总和”：用 df，因为要尽量完整；
# - 但单城市省（上海/北京）的 gdp 在 df 里可能还是 NaN，
#   groupby.sum() 默认把 NaN 当 0，所以它们的“总和”会是 0。
#   这不是 bug，是“数据本身推不出来”的诚实结果，report 里要写清楚。

province_stats = (df.groupby("province").agg(
    城市数 = ("city","count"),
    GDP总量 = ("gdp","sum"),
    GDP均值 = ("gdp","mean"),
    人口总和 = ("resident_pop","sum"),
    面积总和 = ("city_area","sum")
).sort_values("GDP总量",ascending=False)
)

print("=" * 50)
print("各省聚合（按 GDP 总和降序，前 15）：")
print(province_stats.head(15).to_string())

province_stats.to_csv(Path(__file__).with_name("province_stats.csv"),index=True,encoding="utf-8-sig")  # 保存到 CSV，index=True 保留省名列

# ============================================================
# 城市排行榜：用 raw（未填充），剔除 gdp 缺失
# ============================================================

top_10city = (raw.dropna(subset=["gdp"]).nlargest(10,"gdp")[["city","province","gdp","resident_pop"]])
#从原始数据里，找出 GDP 最高的 10 个城市，只保留 4 列，存成 top10_city。

print()
print("GDP 前 10 城市（剔除缺失，不填充）：")
print(top_10city.to_string(index=False))

top_10city.to_csv(Path(__file__).with_name("top10_city.csv"), index=False, encoding="utf-8-sig")