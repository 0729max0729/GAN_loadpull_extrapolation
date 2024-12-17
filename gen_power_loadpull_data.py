import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import colormaps
from skrf.plotting import smith
import os


# 放大器模型函数
def impedance_matching_factor(Z_out, Z_load):
    """
    计算阻抗匹配因子
    Args:
        Z_out (complex): 输出阻抗
        Z_load (complex): 负载阻抗
    Returns:
        float: 匹配因子 M
    """
    return 1 - np.abs((Z_load - Z_out) / (Z_out + Z_load)) ** 2


def amplifier_with_impedance(input_power, Z_in, Z_out, Z_load, G0=10, nonlinearity=[0.1, 0.05]):
    """
    模拟晶体管放大器，考虑输入/输出阻抗对功率的影响
    Args:
        input_power (float): 输入功率
        Z_in (complex): 输入阻抗
        Z_out (complex): 输出阻抗
        Z_load (complex): 负载阻抗
        G0 (float): 初始线性增益
        nonlinearity (list): 非线性系数 [B, C]
    Returns:
        float: 输出功率
    """
    # 匹配因子
    M = impedance_matching_factor(Z_out, Z_load)

    # 动态增益调整
    G = G0 * M

    # 非线性放大
    B, C = nonlinearity
    output_power = G * input_power + B * input_power ** 2 + C * input_power ** 3
    return output_power


# 生成轮廓数据
def generate_contour_data(Z_out, input_power, G0, nonlinearity, radius=1.0, resolution=50):
    """
    生成轮廓数据，用于绘制 Smith Chart Contour
    Args:
        Z_out (complex): 输出阻抗
        input_power (float): 输入功率
        G0 (float): 初始线性增益
        nonlinearity (list): 非线性系数 [B, C]
        radius (float): 负载阻抗的最大幅度
        resolution (int): 角度分辨率
    Returns:
        numpy array: 包含 [Re(Z_load), Im(Z_load), P_out] 的数据
    """
    data = []
    for magnitude in np.linspace(1e-5, radius-1e-5, resolution):  # 幅度扫描
        for angle in np.linspace(0, 2 * np.pi, resolution):  # 相位扫描
            Gamma = magnitude * np.exp(1j * angle)
            Z_load=50*(1+Gamma)/(1-Gamma)  # 负载阻抗
            P_out = amplifier_with_impedance(input_power, Z_in=50 + 0j, Z_out=Z_out, Z_load=Z_load,
                                             G0=G0, nonlinearity=nonlinearity)
            data.append([Gamma.real, Gamma.imag, P_out])
    return np.array(data)


# 保存轮廓数据
def save_contour_data(filename, contour_data):
    """
    保存轮廓数据为 CSV 文件
    Args:
        filename (str): 文件名
        contour_data (numpy array): 包含 [Re(Z_load), Im(Z_load), P_out] 的数据
    """
    df = pd.DataFrame(contour_data, columns=["Re(Z_load)", "Im(Z_load)", "P_out"])
    df.to_csv(filename, index=False)
    print(f"轮廓数据已保存至 '{filename}'")


# 绘制 Smith Chart Contour
def plot_contour_on_smith_chart(filename, output_image):
    """
    绘制 Smith Chart Contour
    Args:
        filename (str): 包含轮廓数据的 CSV 文件名
        output_image (str): 保存绘图的路径
    """
    # 从 CSV 加载数据
    data = pd.read_csv(filename)

    # 绘制 Smith Chart
    fig, ax = plt.subplots(figsize=(2.56, 2.56))
    smith(ax=ax, draw_labels=False)

    # 绘制等值线
    sc = ax.tricontourf(data["Re(Z_load)"], data["Im(Z_load)"], data["P_out"], levels=30, lpha=0.75, cmap="jet")
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
    plt.savefig(output_image)
    plt.close()
    print(f"Smith Chart 已保存至 {output_image}")


# 主程序：生成多张图
if __name__ == "__main__":
    # 参数设置

    input_power = 1.0  # 输入功率
    radius = 1.0  # 负载阻抗幅度最大值
    resolution = 100  # 分辨率
    num_charts = 10  # 生成图的数量

    # 输出目录
    output_data_dir = "test_data"
    output_contour_dir = "test_contour_data"
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_contour_dir, exist_ok=True)
    for i in range(num_charts):
        # 随机增益和非线性参数
        Z_out = np.random.uniform(10, 50) + np.random.uniform(-20, 20)*1j  # 输出阻抗
        G0 = np.random.uniform(5, 15)  # 随机增益
        nonlinearity = [np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5)]  # 随机非线性系数

        # 生成轮廓数据
        contour_data = generate_contour_data(Z_out, input_power, G0, nonlinearity, radius, resolution)

        # 保存数据
        contour_filename = os.path.join(output_data_dir, f"contour_data_{i + 1}.csv")
        save_contour_data(contour_filename, contour_data)

        # 保存图像
        output_image = os.path.join(output_contour_dir, f"smith_chart_{i + 1}.png")
        plot_contour_on_smith_chart(contour_filename, output_image)
