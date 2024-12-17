import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from skrf.plotting import smith

# 生成覆盖整个 Smith Chart 的网格数据
def generate_smith_chart_grid(grid_resolution=100):
    """
    生成覆盖整个 Smith Chart 的反射系数网格点
    Args:
        grid_resolution (int): 网格分辨率（每个轴上的点数）
    Returns:
        gamma_real, gamma_imag (ndarray): 网格上的反射系数实部和虚部
    """
    x = np.linspace(-1, 1, grid_resolution)
    y = np.linspace(-1, 1, grid_resolution)
    x, y = np.meshgrid(x, y)
    mask = np.sqrt(x**2 + y**2) <= 1  # 只保留单位圆内的点
    gamma_real = x[mask]
    gamma_imag = y[mask]
    return gamma_real, gamma_imag

# 随机化的增益函数
def randomized_gain_function(magnitude, phase, base_gain=20, variability=10):
    """
    随机微调的增益函数
    Args:
        magnitude (ndarray): 反射系数的幅度
        phase (ndarray): 反射系数的相位
        base_gain (float): 基础增益值
        variability (float): 增益的随机波动范围
    Returns:
        gain (ndarray): 计算后的增益值
    """
    alpha = np.random.uniform(0, 0)  # 控制 exp 函数的幅度
    beta = np.random.uniform(0, 3)  # 控制 cos 函数的幅度
    offset = np.random.uniform(-10, 10)  # 基础增益的随机偏移量

    # 增益函数
    gain = base_gain + offset + variability * np.exp(-alpha * magnitude) * np.cos(beta * phase)
    return gain

# 生成 Load-Pull 数据
def generate_load_pull_data(output_file, gamma_real, gamma_imag, base_gain=20, variability=10):
    """
    计算网格上的增益并保存到文件
    Args:
        output_file (str): 保存的文件路径
        gamma_real, gamma_imag (ndarray): 网格上的反射系数
        base_gain (float): 基础增益值
        variability (float): 增益的随机波动范围
    Returns:
        data (pd.DataFrame): 包含生成的 load-pull 数据
    """
    magnitude = np.sqrt(gamma_real**2 + gamma_imag**2)
    phase = np.arctan2(gamma_imag, gamma_real)
    gain = randomized_gain_function(magnitude, phase, base_gain, variability)

    data = pd.DataFrame({
        "Gamma_Real": gamma_real,
        "Gamma_Imag": gamma_imag,
        "Magnitude": magnitude,
        "Phase (radians)": phase,
        "Gain (dB)": gain
    })
    data.to_csv(output_file, index=False)
    print(f"Load-pull data saved to {output_file}")
    return data

# 在 Smith Chart 上绘制 Load Pull Contour
def plot_load_pull_contour(data, levels=10, output_file=None):
    """
    在 Smith Chart 上绘制 Load Pull Contour
    Args:
        data (pd.DataFrame): 包含 load-pull 数据的 DataFrame
        levels (int): 等值线层级数量
        output_file (str, optional): 如果提供，则将图保存到此文件路径
    """
    gamma_real = data["Gamma_Real"].values
    gamma_imag = data["Gamma_Imag"].values
    gain = data["Gain (dB)"].values

    # 创建 Smith Chart
    fig, ax = plt.subplots(figsize=(2.56, 2.56))
    smith(ax=ax, draw_labels=False)

    # 绘制等值线
    contour = ax.tricontourf(gamma_real, gamma_imag, gain, levels=levels,  alpha=0.75)

    # 设置图形属性
    ax.axis("scaled")  # 确保比例正确
    ax.set_xlim([-1, 1])  # 让 Smith Chart 填满整个轴
    ax.set_ylim([-1, 1])

    # 删除四周的多余轴和空白
    ax.set_xticks([])  # 删除 x 轴刻度
    ax.set_yticks([])  # 删除 y 轴刻度
    ax.set_xlabel("")  # 删除 x 轴标签
    ax.set_ylabel("")  # 删除 y 轴标签
    ax.spines['top'].set_visible(False)  # 删除顶部边框
    ax.spines['right'].set_visible(False)  # 删除右侧边框
    ax.spines['left'].set_visible(False)  # 删除左侧边框
    ax.spines['bottom'].set_visible(False)  # 删除底部边框

    # 保存图像
    if output_file:
        plt.savefig(output_file)
        print(f"Contour plot saved to {output_file}")
    plt.close()

# 主函数：批量生成数据并保存
if __name__ == "__main__":
    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    # 设置扫描的网格分辨率和文件数量
    grid_resolution = 100  # Smith Chart 网格分辨率
    num_files = 5000  # 生成的文件数量

    for i in range(1, num_files + 1):
        # 输出文件路径
        data_file = f"data1/load_pull_data_{i}.csv"
        plot_file = f"data/load_pull_contour_{i}.png"

        # 生成覆盖整个 Smith Chart 的网格数据
        gamma_real, gamma_imag = generate_smith_chart_grid(grid_resolution=grid_resolution)

        # 生成 Load-Pull 数据
        data = generate_load_pull_data(data_file, gamma_real, gamma_imag, base_gain=20, variability=10)

        # 绘制并保存 Load Pull Contour
        plot_load_pull_contour(data, levels=20, output_file=plot_file)
