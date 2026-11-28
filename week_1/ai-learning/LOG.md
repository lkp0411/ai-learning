# 第 1 周日志（Day 1 – Day 7）

时间：2026-11-18 ~ 2026-11-23
环境：Windows 11 · Miniconda `ai` 环境（Python 3.11.18）· VS Code · NVIDIA GeForce RTX 5060 Laptop

---

## Day 1 · 环境搭建（11-18）

**目标**：能在一个配好 GPU 的 Python 环境里跑出 `torch.cuda.is_available() == True`。

做的事：
- 装 VS Code 扩展：Python、Pylance、Jupyter、中文语言包
- 关掉「受限模式」（否则扩展全部禁用，F5 不工作）
- 界面中文不生效 → 语言包版本和 VS Code 版本不匹配，最后用启动参数 `--locale=zh-cn` 解决
- 装 Miniconda（清华镜像），建 `conda create -n ai python=3.11`
- 装 PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

关键结论：
- **必须 cu128**。50 系显卡是 `sm_120` 计算能力，旧版 torch（cu118/cu121）会报 `no kernel image is available`
- 装包永远用 `python -m pip install`，否则会装到别的 Python 里

验证结果：

```
torch          : 2.11.0+cu128
cuda           : 12.8  cudnn: 91900
is_available   : True
GPU            : NVIDIA GeForce RTX 5060 Laptop GPU
compute        : sm_120
4000x4000 matmul: 0.166 s (771 GFLOPS)
```

产物：`docs/gpu_check.png`

---

## Day 2 · pandas 基础（11-19）

**目标**：能读 CSV、看结构、过滤、排序、分组。

做的事：
- `read_csv.py`：读 CSV、写 CSV（`to_csv(index=False)`）
- `hello_stats.py`：手写 `mean` / `std`，理解 pandas 之前的世界
- 建的笔记文件 `11.19` 没扩展名，被 VS Code 当成 Python，满屏红波浪线 → 改名 `11.19.md`

踩的坑：
- `pd.DataFrame({"a":[1,2,3]}, {"b":["x","y","z"]})` —— 第二个位置参数是 **index**，不是第二列，报 `Length of values does not match length of index`
- `def std(xs): mean = means(xs)` —— 拼写错 + 变量名遮蔽函数名（`mean = mean(...)` 会 `UnboundLocalError`）

---

## Day 3 · 命令行工具（11-19）

**目标**：写一个能用参数调用的脚本：`python tool.py --input ... --output ... --top N`。

做的事：`tool.py`，用 `argparse` 解析三个参数，读 CSV → 按某列降序 → 取前 N 行 → 写新 CSV。

踩的坑：
- **脚本执行了但什么也没输出** —— 根因是只有 `def main()` 的定义，没有 `if __name__ == "__main__": main()`。Python 只是"定义了一个函数"，从来没调用它。
- 命令行末尾多打了反引号 `` ` `` —— PowerShell 里那是续行符，会卡在 `>>` 提示符，命令根本没提交
- `SORT_COL = "a"` 写死，喂真实数据会报"输入列中没有列 'a'"

---

## Day 4 · 数据分析 + 可视化（11-20）

**目标**：找一份真实数据，走完 读入 → 清洗 → 统计 → 画图 → 写报告。

数据：`gdp.csv`，332 行 × 7 列，26 个省的 332 个城市（城市经济数据）。

做的事：
- `make_missing.py`：人为注入缺失值（随机挖 4%，`builtup_area` 再挖 10%），并塞几个假缺失 `N/A`、`-`
- `analyze.py`：处理缺失值、算省份汇总、出城市 GDP 榜
- `plot.py`：3 张图（省份 GDP 柱状图、城市 Top10、人口-GDP 散点图）
- `report.md`：数据来源、清洗过程、图说明、局限

**这周写出的最重要的一个 bug**：按省分组中位数填 GDP，上海从 24964.99 被填成了 **1247.2**（0.05 倍）。

原因：上海的 `province` 和 `city` 同名，`groupby("province")` 之后组里只有它自己一行，而这一行正好是缺失值，`transform("median")` 就返回 `NaN`，再落进"全局中位数"的兜底逻辑。北京同理。

修法：先用 `transform("size")` 算出每个省有多少个城市，只对 `>= 3` 个城市的省做组内填充，单城市省保留 `NaN`，后面显式 `dropna` 再做排行榜。

对比数据（填充前后的失真）：

| 指标 | 完整原始数据 | 我的填充方案 |
|---|---|---|
| 上海 GDP | 24964.99 | 1247.2 ❌ |
| 神农架 GDP | 20.95 | 1388.5 ❌ |
| GDP 标准差 | 2534.45 | 2484.04 |
| GDP×人口 相关系数 | 0.8420 | 0.6943 ❌ |

结论：**大部分缺失值不该填**。填了之后相关系数从 0.84 掉到 0.69，这不是"补全"，这是把数据改坏了。

---

## Day 5 · Git + GitHub（11-21）

**目标**：把项目传上 GitHub，并且自己能复述每一步在干什么。

做的事：
- 装 Git for Windows（2.56.1），配 `user.name` / `user.email`
- 建仓库 `lkp0411/ai-learning`（public）
- `git init -b main` → `git add -A` → `git commit` → `git push -u origin main`

踩的坑：
- **新装的 git 在老终端里认不出来** —— PATH 是进程启动时的快照，装完 git 必须**新开一个终端**
- `git commit -m "中文"` 存进去变乱码 `Day 4: 鍩庡競 GDP 鏁版嵁...` —— PowerShell 5.1 用 GBK 传参数。改用 `git commit -F 提交信息.txt`（文件存 UTF-8）
- 用户名一开始记错了（`lpk0411` vs `lkp0411`），查 GitHub API 一直 404

最终提交：`ea30d89` — `Day 4:GDP data cleaning and visualization`，12 个文件。

对 Git 的理解（用寄快递打比方）：
- 工作区 = 桌面上摊着的东西
- `git add` = 装进箱子
- `git commit` = 封箱 + 贴单子（单子上写着这次改了什么）
- `git push` = 送到快递站（远端）
- `.gitignore` = 清单上写明"这几样不用寄"（可再生的、个人配置、密码密钥）

---

## Day 6 · GPU 上的线性回归（11-23）

**目标**：在 GPU 上跑通一个完整的训练循环。

做的事：`train_linear.py`
- 造数据：`y = 3x + 2 + 噪声`
- `nn.Linear(1,1)` + `MSELoss` + `SGD(lr=0.01)`
- 训练 100 轮，每轮五步：前向 → 算损失 → 梯度清零 → 反向传播 → 更新参数

**结果：跑通了，但训练发散了。**

```
当前使用的设备:cuda
第10轮，损失:2085.1587
第50轮，损失:393792.1875
第100轮，损失:277069376.0000

真实的参数：w=3.000，b=2.000
学到的参数：w=-1744.9521，b=-0.9646
```

原因：`X = torch.randn(100,1) * 10` 数值太大（±30），学习率 0.01 对它来说步子太大，一步跨过谷底越飞越远。

对照实验结论：

| 方案 | 第 100 轮 loss | 学到的 w | 学到的 b |
|---|---|---|---|
| lr=0.01（原样） | 1.38e11 | -39871 | -140 |
| lr=0.002 + 500 轮 | 3.48 | 2.9775 | 1.9474 |
| 标准化 X + lr=0.1 | 3.34 | 2.9762 | 2.3194 |

**待办**：把 `lr` 改成 0.002、轮数改 500，重跑确认收敛。

---

## Day 7 · 盲写检验 + 复盘（11-23）

做的事：
- 不看资料重写一遍线性回归训练循环（五步顺序：前向 → loss → zero_grad → backward → step）
- 整理目录：`scripts/` 三个脚本、`data/` 原始数据、`outputs/` 产物、`docs/` 截图
- 汇总 `LOG.md`（本文）
- 写 `week1-review.md` 复盘

---

## 本周验收

| 验收项 | 状态 | 证据 |
|---|---|---|
| `torch.cuda.is_available() == True` | ✅ | `docs/gpu_check.png`、Day 1 |
| GitHub 仓库 + ≥3 个 .py + README | ✅ | `lkp0411/ai-learning`，`scripts/` 下 3 个脚本 |
| 命令行工具 `python tool.py --input ...` | ✅ | Day 3，`11.19/tool.py`（未进仓库） |
| 数据分析脚本 + 3 张图 | ✅ | `outputs/fig1~fig3.png` |
| GPU 上跑通线性回归训练循环 | ⚠️ 跑通但未收敛 | 见 Day 6，学习率待调 |
| 盲写 + 7 条每日复盘 | ✅ | 本文件 |

---

## 下周要做

1. 修好 `train_linear.py` 的学习率，让它真的收敛到 w≈3、b≈2
2. 搞懂 `nn.Module`：自己写一个模型类（而不是只用 `nn.Linear`）
3. 理解学习率、数据尺度、梯度大小三者的关系
4. 补上 `docs/gpu_check.png`（现在仓库里还没有）
