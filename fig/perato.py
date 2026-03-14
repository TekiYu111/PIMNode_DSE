# perato.py
import matplotlib
matplotlib.use('Agg')  # 无头环境使用
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 加载中文字体
my_font = font_manager.FontProperties(fname='SimHei.ttf')

# 设置随机种子
np.random.seed(42)

# 横坐标（优化目标1）
x = np.linspace(0.1, 1.0, 50)

# 理想帕累托前沿（红色虚线）
pareto_front = 1.0 / x

# 样本解（蓝色点），随机分布更宽
f1_samples = x + 0.2 * (np.random.rand(50) - 0.5)  # 横向偏离 ±0.1
f2_samples = pareto_front + 1.0 * (np.random.rand(50) - 0.5)  # 纵向偏离 ±0.5

# 创建图形
plt.figure(figsize=(6,5))

# 蓝色样本点
plt.scatter(f1_samples, f2_samples, c='blue', label='样本解')

# 红色虚线帕累托前沿
plt.plot(x, pareto_front, 'r--', linewidth=2, label='帕累托前沿')

# 中文标签
plt.xlabel('优化目标1', fontproperties=my_font, fontsize=12)
plt.ylabel('优化目标2', fontproperties=my_font, fontsize=12)
plt.title('帕累托前沿示意图', fontproperties=my_font, fontsize=14)

# 图例
plt.legend(prop=my_font)

# 网格
plt.grid(True)

# 保存为 PNG
plt.savefig("pareto_plot.png", dpi=300, bbox_inches='tight')
print("图已保存为 pareto_plot.png")
