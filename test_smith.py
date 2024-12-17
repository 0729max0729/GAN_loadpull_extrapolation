import numpy as np

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
    return M
def find_max_gain(Z_out, input_power, G0, nonlinearity, resolution=100):
    """
    找到在指定输出阻抗 Z_out 下，最大增益对应的 Z_load
    Args:
        Z_out (complex): 输出阻抗
        input_power (float): 输入功率
        G0 (float): 初始线性增益
        nonlinearity (list): 非线性系数 [B, C]
        resolution (int): 扫描的分辨率
    Returns:
        Z_load_max_gain (complex): 最大增益对应的负载阻抗
        max_gain (float): 最大增益值
    """
    max_gain = -np.inf
    Z_load_max_gain = None

    # 扫描 Z_load 的幅度和相位
    for magnitude in np.linspace(1e-5, 1-1e-5, resolution):  # 幅度范围 [0, 1]
        for angle in np.linspace(0, 2 * np.pi, resolution):  # 相位范围 [0, 2π]
            Gamma = magnitude * np.exp(1j * angle)
            Z_load = 50 * ((1 + Gamma) / (1 - Gamma) ) # 负载阻抗
            gain = amplifier_with_impedance(input_power, Z_in=50 + 0j, Z_out=Z_out, Z_load=Z_load,
                                             G0=G0, nonlinearity=nonlinearity)
            if gain > max_gain:
                max_gain = gain
                Z_load_max_gain = Z_load

    return Z_load_max_gain, max_gain


# 测试代码
Z_out = 50  # 输出阻抗
input_power = 1.0  # 输入功率
G0 = 10  # 初始线性增益
nonlinearity = [0.1, -0.05]  # 非线性系数

Z_load_max_gain, max_gain = find_max_gain(Z_out, input_power, G0, nonlinearity)
print(f"最大增益对应的 Z_load: {Z_load_max_gain}")
print(f"最大增益值: {max_gain}")
