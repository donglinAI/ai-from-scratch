
"""
xor_solution_vis.py —— XOR 问题解决绘图
"""

import numpy as np             
import matplotlib    
matplotlib.use("Agg")           # 使用 Agg 后端：不弹窗，直接保存图片文件
import matplotlib.pyplot as plt 

def plot_xor_solution(X, y, predict, save_path="xor_solution.png"):

    X = np.asarray(X, dtype=float) # 将输入统一转成 numpy float 数组，兼容 list / torch tensor 等输入
    y = np.asarray(y, dtype=float).ravel() # 将标签转成一维 numpy 数组

    color0, color1 = "#9BBBF4", "#EA6668" # 0（蓝色），（1）红色
      
    xs = np.linspace(-0.3, 1.3, 250) # 生成网格坐标：x 方向从 -0.3 到 1.3 均匀取 250 个点
    ys = np.linspace(-0.3, 1.3, 250)
    
    xx, yy = np.meshgrid(xs, ys) # 用两组一维坐标生成二维网格坐标矩阵 xx, yy
    grid = np.c_[xx.ravel(), yy.ravel()] # 将网格展平为 (N, 2) 的输入矩阵，方便一次性预测

    z = np.asarray(predict(grid), dtype=float).reshape(xx.shape) # 调用外部 predict 对每个网格点预测，再 reshape 回网格形状
    probs = np.asarray(predict(X), dtype=float).ravel()
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)  # 画布尺寸和分辨率
    # 用 contourf 填充背景：颜色深浅表示模型预测概率（蓝=0 类、红=1 类）
    ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 50), cmap="RdBu", alpha=0.55)
    cs = ax.contour(xx, yy, z, levels=[0.5], colors="#1A1B1C", linewidths=2) # 画 p=0.5 的等高线，即决策边界线
    ax.clabel(cs, fmt="p=0.5", fontsize=10) # 在边界线上标注其概率值

    for i in range(len(y)):
        label = int(y[i])
        color = color0 if label == 0 else color1
        # 判断预测是否正确：预测值 > 0.5 判为 1，再与真实标签比较
        correct = int(probs[i] > 0.5) == label
        # 画散点：预测正确用黑色描边，错误用橙色描边
        ax.scatter(X[i, 0], X[i, 1], s=500, c=color,
                   edgecolors=("black" if correct else "#FFB800"),
                   linewidths=3, zorder=5)
        # 在点右上方标注真实标签，如 XOR(0,1)=1
        ax.text(X[i, 0] + 0.03, X[i, 1] + 0.03,
                f"XOR({int(X[i,0])},{int(X[i,1])})={label}",
                fontsize=11, zorder=6)
    ax.set_xlim(-0.5, 1.5) # 设置 x 轴显示范围
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]) # 设置 x 轴刻度为 0 和 1
    ax.set_yticks([0, 1])
    ax.set_xlabel("x1") # 坐标轴名称
    ax.set_ylabel("x2") 

    # 设置标题：决策边界 + 训练集准确率（英文）
    ax.set_title("XOR Solution", fontsize=16, fontweight="bold", pad=12)
    ax.grid(True, ls=":", color="#D1D5DB", alpha=0.7) # 添加浅色虚线网格辅助读数
    
    fig.tight_layout() # 自动调整子图布局
    fig.savefig(save_path, bbox_inches="tight", facecolor="white") # 保存图片到指定路径：白底、紧凑边框
    plt.close(fig) # 关闭画布，释放内存

