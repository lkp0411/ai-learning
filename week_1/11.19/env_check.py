"""env_check.py — 体检：现在这个 Python 到底是哪个、装了什么包。

用法：
    在 VS Code 里打开本文件，按 F5（或 Shift+Enter / 右上角 ▶）运行。
    看到的结果就是"编辑器选中的解释器"的实际情况。

命令行也可以：
    python env_check.py
"""

import sys
import importlib

print("=" * 46)
print("Python 版本 :", sys.version.split()[0])
print("解释器路径 :", sys.executable)
print("=" * 46)

# 想查哪些包，就往这个列表里加
packages = ["pandas", "numpy", "matplotlib", "openpyxl", "torch"]

for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"  [OK]   {name:<12} {version}")
    except ImportError:
        print(f"  [缺]   {name:<12} 未安装")

print("=" * 46)
print("如果这里显示 [缺]，但你以为装过 → 装到别的 Python 里了。")
print("正确装法：python -m pip install 包名   （一定要带 -m pip）")
