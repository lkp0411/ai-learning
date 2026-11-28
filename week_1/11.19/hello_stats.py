import random

def mean(xs):   #求平均值
    return sum(xs) / len(xs)

def std(xs):   #求标准差
    m = mean(xs)          # 注意别写成 mean = mean(xs)，会把函数名覆盖掉
    variance = sum((x - m) ** 2 for x in xs) / len(xs)
    return variance ** 0.5

xs = [random.random() for _ in range(100)]
print(mean(xs))
print(std(xs))

import random
import math

def mean(xs):
    if not xs:
        raise ValueError("xs 不能为空")
    return sum(xs) / len(xs)

def std(xs):
    m = mean(xs)
    variance = sum((x - m) ** 2 for x in xs) / len(xs)  # 总体标准差
    return math.sqrt(variance)

xs = [random.random() for _ in range(100)]
print(f"mean = {mean(xs):.6f}")
print(f"std  = {std(xs):.6f}")