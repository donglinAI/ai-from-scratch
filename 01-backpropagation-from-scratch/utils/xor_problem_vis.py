"""
xor_problem_vis.py —— XOR 问题绘图
"""

import numpy as np             
import matplotlib    
matplotlib.use("Agg")           # 使用 Agg 后端：不弹窗，直接保存图片文件
import matplotlib.pyplot as plt 


def plot_xor_problem(X, y, save_path="xor_problem.png"):

    X = np.asarray(X, dtype=float) # 将输入统一转成 numpy float 数组，兼容 list / torch tensor 等输入
    y = np.asarray(y, dtype=float).ravel() # 将标签转成一维 numpy 数组

    color0, color1 = "#9BBBF4", "#EA6668" # 0（蓝色），（1）红色
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150) # 画布尺寸和分辨率
    for i in range(len(y)):
        label = int(y[i])
        color = color0 if label == 0 else color1
        # 画散点：s 控制点大小，边缘白色描边，zorder 保证点在网格之上
        ax.scatter(X[i, 0], X[i, 1], s=600, c=color,
                   edgecolors="white", linewidths=2.5, zorder=5)
        # 在点右上方标注文字，如 XOR(0, 1) = 1
        ax.text(X[i, 0] + 0.04, X[i, 1] + 0.04,
                f"XOR({int(X[i,0])}, {int(X[i,1])}) = {label}",
                fontsize=13, fontweight="bold")
    
    ax.set_xlim(-0.5, 1.5) # 设置 x 轴显示范围
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]) # 设置 x 轴刻度为 0 和 1
    ax.set_yticks([0, 1])
    ax.set_xlabel("x1") # 坐标轴名称
    ax.set_ylabel("x2") 

    ax.set_title("XOR Problem", fontsize=16, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11, frameon=False) # 显示图例
    ax.grid(True, ls=":", color="#D1D5DB", alpha=0.7) # 添加浅色虚线网格辅助读数

    fig.tight_layout() # 自动调整子图布局，避免元素重叠
    fig.savefig(save_path, bbox_inches="tight", facecolor="white") # 保存图片到指定路径：白底、紧凑边框
    plt.close(fig) # 关闭画布，释放内存

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32) 
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    plot_xor_problem(X, y, save_path="xor_problem.png")