"""tool.py — 读 CSV，按某列降序排序，取前 N 行，写出新 CSV。

用法：
    python tool.py --input data.csv --output result.csv --top 3
"""

import argparse
import logging
import pandas as pd
import sys

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SORT_COL = "a"   # 排序列先固定；想做成参数的话，见文件末尾的“可选扩展”

def parse_args():
    parser = argparse.ArgumentParser(
        description="读 CSV，按指定列降序排序，取前 N 行，写出新 CSV"
    )
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument(
        "--output", default="result.csv", help="输出 CSV 路径（默认 result.csv）"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="取前 N 行（默认 10）"
    )
    return parser.parse_args()

def main():
    args = parse_args()  # 解析命令行参数

    # ① 读文件：只捕获“文件不存在”这一种预期异常
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        logger.error("找不到输入文件 %s，请检查路径", args.input)
        sys.exit(1)

    logger.info("读取%s行数据", len(df))

    # ② 排序列不存在时给友好提示，而不是让 pandas 甩 KeyError
    if SORT_COL not in df.columns:
        logger.error("输入列中没有列%r,现有列: %s ", SORT_COL, list(df.columns))
        sys.exit(1)

    # ③ 排序 → 取前 N → 写文件
    result = df.sort_values(SORT_COL,ascending=False).head(args.top)
    result.to_csv(args.output,index=False)

    logger.info("已写出 %s（%d 行）", args.output, len(result))


if __name__ == "__main__":
    main()