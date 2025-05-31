import matplotlib.pyplot as plt
import re
from result.opt_125m import hessian_trace
import numpy as np

# 用于存储每层的数据
layer_data = {}
# 用于存储所有的矩阵名称
matrix_names = set()

# 遍历字典，提取层数和矩阵名称
for key, value in hessian_trace.items():
    match = re.search(r'layers\.(\d+)', key)
    if match:
        layer = int(match.group(1))
        matrix_name = key.split(f'layers.{layer}.')[1]
        if layer not in layer_data:
            layer_data[layer] = {}
        layer_data[layer][matrix_name] = value
        matrix_names.add(matrix_name)

matrix_names = sorted(matrix_names)

# 提取所有值用于统计
all_values = [val for layer in layer_data.values() for val in layer.values()]

# 计算95%分位数作为裁剪阈值
clip_threshold = np.percentile(all_values, 95)

# 对数据进行裁剪和标准化
normalized_layer_data = {}
for layer, data in layer_data.items():
    # 裁剪最大值
    clipped_data = {name: min(val, clip_threshold) for name, val in data.items()}
    
    # 计算均值和标准差（保留两位小数）
    mean_val = round(np.mean(list(clipped_data.values())), 2)
    std_val = round(np.std(list(clipped_data.values())), 2)
    
    # 避免除以零
    if std_val == 0:
        normalized_data = {name: 0.0 for name in clipped_data}
    else:
        normalized_data = {name: (val - mean_val) / std_val for name, val in clipped_data.items()}
    
    normalized_layer_data[layer] = normalized_data

# 定义颜色列表
colors = plt.cm.tab20.colors

# 设置图形大小
plt.figure(figsize=(14, 8))

# 绘制柱状图
bar_width = 0.8 / len(matrix_names)
for i, matrix_name in enumerate(matrix_names):
    values = []
    for layer in sorted(normalized_layer_data.keys()):
        values.append(normalized_layer_data[layer].get(matrix_name, 0))
    
    # 使用半透明填充和深色边框增强可读性
    plt.bar(
        [j + i * bar_width for j in range(len(normalized_layer_data))], 
        values, 
        width=bar_width, 
        color=colors[i % len(colors)], 
        alpha=0.7, 
        edgecolor='black', 
        linewidth=0.5,
        label=matrix_name if i < 10 else ""  # 限制图例数量，避免拥挤
    )

# 设置图形标签和标题
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Standardized Value (Z-score)', fontsize=12)
plt.title('Standardized Hessian Trace Values per Layer (Clipped at 95th Percentile)', fontsize=14)
plt.xticks([j + bar_width * (len(matrix_names) - 1) / 2 for j in range(len(normalized_layer_data))], sorted(normalized_layer_data.keys()))

# 添加网格线增强可读性
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整图例位置和样式
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=9)

# 添加标注解释裁剪操作
plt.figtext(0.5, 0.01, 
            f'Note: Values above the 95th percentile ({clip_threshold:.2e}) have been clipped for visualization.', 
            ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为底部的图例和注释留出空间

# 显示图形
plt.show()