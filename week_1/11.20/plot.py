"""plot.py — 读同目录下的 CSV 和 gdp.csv，画 3 张图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 脚本所在目录 = 11.20/
BASE = Path(__file__).parent

# 读 analyze.py 存下来的两个 CSV（都在 11.20/ 根目录）
province_csv = pd.read_csv(BASE / "province_stats.csv", index_col=0)
top10_city = pd.read_csv(BASE / "top10_city.csv")

# 散点图用未填充的原始数据
raw = pd.read_csv(BASE / "gdp.csv")
raw = raw.replace(["-", "暂无", "N/A", "null"], pd.NA)
for col in ["gdp","resident_pop"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")  # 转为数值型，无法转换的变为 NaN

# ---------- 图 1：柱状图，GDP 总量前 15 省份 ----------

top15 = province_csv.head(15)

plt.figure(figsize=(12, 6))
plt.bar(top15.index, top15["GDP总量"], color="orange")#plt.bar(x, y)：画柱状图；top15.index：x 轴，是省份名（因为 province_stats 读回来时 index_col=0，索引就是省份）；top15["GDP总量"]：y 轴，是每个省的 GDP 总量；
plt.title(" GDP 总量前 15")
plt.xlabel("省份")
plt.ylabel("GDP总量（亿元）")
plt.xticks(rotation=45,ha="right")
"""
调 x 轴刻度标签的显示方式：

rotation=45：标签旋转 45 度。省份名两三个字，不转会互相压；

ha="right"：水平对齐方式 = 右对齐（horizontal alignment）。旋转后如果还左对齐，标签会往右偏，压到下一个刻度；右对齐让标签的右上角对准刻度点，看起来整齐。
"""
plt.tight_layout()#自动调整子图参数，防止标签被裁掉。
plt.savefig(BASE/"fig1_bar_gdp_by_province.png",dpi=150)
plt.close()

# ---------- 图 2：折线图，GDP 前 10 城市 ----------
plt.figure(figsize=(10, 6))
plt.plot(top10_city["city"], top10_city["gdp"], marker="o",linewidth=2, color="green")
plt.title("GDP 前 10 城市")
plt.xlabel("城市")
plt.ylabel("GDP（亿元）")
plt.xticks(rotation=45,ha="right")
plt.grid(True,alpha=0.3)
"""
加网格线。

True：显示网格；

alpha=0.3：透明度 0.3（0 全透明，1 不透明）。0.3 是很淡的灰色网格，帮助读数但不抢眼。
"""
plt.tight_layout()
plt.savefig(BASE/"fig2_line_gdp_top10_cities.png",dpi=150)
plt.close()

# ---------- 图 3：散点图，常住人口 vs GDP ----------
plot_df = raw.dropna(subset=["gdp","resident_pop"]) # 只取两列，且去掉缺失值

plt.figure(figsize=(9, 6))
plt.scatter(plot_df["resident_pop"],plot_df["gdp"],alpha=0.5,s=20,edgecolors="none")
"""
alpha=0.5：点半透明。散点图点多了会重叠，半透明能看出“哪里点密集”；

s=20：点的大小（size）。默认 36，20 小一点，317 个点不会糊成一团；

edgecolors="none"：去掉点的边框。默认每个点有黑色边框，点多了显脏，去掉更干净。
"""
for _,row in top10_city.head(5).iterrows():  # 只标注前 5 个城市
    plt.annotate(row["city"],(row["resident_pop"],row["gdp"]),fontsize=9,xytext=(5,5),textcoords="offset points")
"""
plt.annotate(text, xy, ...)：在指定坐标处加文字标注。

row["city"]：要显示的文字，比如 "深圳"；

(row["resident_pop"], row["gdp"])：标注指向的坐标，就是那个点的位置；

fontsize=9：字号 9，比默认小一点，不抢眼；

xytext=(5, 5)：文字相对坐标点的偏移量；

textcoords="offset points"：偏移量的单位是“点”（point），(5, 5) 表示往右 5 点、往上 5 点。
"""
plt.title("常住人口 vs GDP")
plt.xlabel("常住人口（万人）")
plt.ylabel("GDP（亿元）")
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(BASE/"fig3_scatter_gdp_vs_resident_pop.png",dpi=150)
plt.close()

print("三张图已保存到 11.20/ 目录下：fig1_bar_gdp_by_province.png、fig2_line_gdp_top10_cities.png、fig3_scatter_gdp_vs_resident_pop.png")