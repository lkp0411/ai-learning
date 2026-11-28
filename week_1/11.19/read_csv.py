import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

df.to_csv("data.csv", index=False)  # 将 DataFrame 保存为 CSV 文件，不保存索引

df2 = pd.read_csv("data.csv")  # 从 CSV 文件中读取数据
print(df2.head())  # 打印前几行数据
print(df2.dtypes)  # 打印每列的数据类型
print(df2.describe())  # 打印数值列的统计信息
print(df2.info())  # 打印 DataFrame 的信息，包括行数、列数和数据类型
print(df2.columns)  # 打印列名
print(len(df2))  # 打印行数