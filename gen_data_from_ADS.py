import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# 讀取資料
file_path = 'ADS_loadpull_data/load_pull_PAE_sim_data_90nm_sweep_source_freq.csv'
data = pd.read_csv(file_path)

# 清理欄位名稱
data.columns = ["RFfreq","Vlow", "M", "real_indexs22", "imag_indexs22","imag_indexs11", "real_indexs11", "PAE"]

# 確保資料為數值型態
data = data.apply(pd.to_numeric, errors='coerce')

# 創建一個目錄來存儲圖像
output_dir = "ADS_smith_chart_plots"
os.makedirs(output_dir, exist_ok=True)

# 定義函數來保存 Smith 圖
def save_pae_smithchart(data, Vlow, M, real_indexs22, imag_indexs22,RFfreq, output_path):
    # 篩選符合條件的數據
    filtered_data = data[(data["Vlow"] == Vlow) & (data["imag_indexs22"] == imag_indexs22) & (data["M"] == M) & (data["real_indexs22"] == real_indexs22)&(data["RFfreq"] == RFfreq)]

    # 提取 REAL, IMAG 和 PAE 值
    real = filtered_data["real_indexs11"].values
    imag = filtered_data["imag_indexs11"].values
    pae = filtered_data["PAE"].values

    # 計算極坐標 (幅值和角度)
    magnitude = np.sqrt(real ** 2 + imag ** 2)
    angle = np.arctan2(imag, real)

    # 建立極坐標網格
    mag_unique = np.linspace(0, 1, 100)
    angle_unique = np.linspace(-np.pi, np.pi, 200)
    mag_grid, angle_grid = np.meshgrid(mag_unique, angle_unique)

    # 將極坐標轉回直角坐標進行插值
    real_grid = mag_grid * np.cos(angle_grid)
    imag_grid = mag_grid * np.sin(angle_grid)
    points = np.column_stack((real, imag))
    pae_grid = griddata(points, pae, (real_grid, imag_grid), method='linear')

    # 繪製 Smith 圖
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2.56, 2.56))
    contour = ax.contourf(angle_grid, mag_grid, pae_grid, cmap='jet', levels=50)

    # 設置圖形屬性
    ax.axis("scaled")  # 確保比例正確
    ax.set_xticks([])  # 刪除 x 軸刻度
    ax.set_yticks([])  # 刪除 y 軸刻度
    ax.set_xlabel("")  # 刪除 x 軸標籤
    ax.set_ylabel("")  # 刪除 y 軸標籤
    ax.set_ylim(0, 1)  # 限制幅值範圍到 Smith 圖

    # 保存圖像
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# 獲取所有的 Vlow, numf, M 的唯一值
unique_vlow = data["Vlow"].unique()
unique_real_indexs22 = data["real_indexs22"].unique()
unique_imag_indexs22 = data["imag_indexs22"].unique()
unique_m = data["M"].unique()
unique_RFfreq=data["RFfreq"].unique()
# 遍歷所有組合並保存圖像
for vlow in unique_vlow:
    for real_indexs22 in unique_real_indexs22:
        for imag_indexs22 in unique_imag_indexs22:
            for m in unique_m:
                for RFfreq in unique_RFfreq:
                    if (abs(real_indexs22) < 1) & (abs(imag_indexs22) <1):
                        output_path = os.path.join(output_dir, f"smithchart_vlow{vlow}_m{m}_real_indexs22{real_indexs22}_imag_indexs22{imag_indexs22}_freq{RFfreq}.png")
                        save_pae_smithchart(data, vlow, m, real_indexs22, imag_indexs22,RFfreq, output_path)
                        print(vlow, m, real_indexs22, imag_indexs22,RFfreq)

print(f"所有 Smith 圖已存儲於 {output_dir}")
