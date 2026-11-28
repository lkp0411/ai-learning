import pandas as pd

INPUT = "data.csv"
OUTPUT = "result.csv"
TOP = 2
SORT_COL = "a"

df = pd.read_csv(INPUT)  # 从 CSV 文件中读取数据
df = df.sort_values(by=SORT_COL, ascending=False)  # 按指定列降序排序
df = df.head(TOP)  # 取前 TOP 行
df.to_csv(OUTPUT, index=False)  # 将结果保存为 CSV 文件，不保存
print("done")   