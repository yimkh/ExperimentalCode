import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

file_path = "data.xlsx"
df = pd.read_excel(file_path)

df.rename(columns={
    "光伏有功功率": "PV_Active_Power",
    "温度": "Temperature",
    "湿度": "Humidity",
    "全球水平辐射": "Global_Horizontal_Irradiance",
    "散射水平辐射": "Diffuse_Horizontal_Irradiance",
    "风向": "Wind_Direction",
    "日降水量": "Daily_Precipitation",
    "倾斜面全球辐射": "Tilted_Global_Irradiance",
    "倾斜面散射辐射": "Tilted_Diffuse_Irradiance"
}, inplace=True)

features = [
    "PV_Active_Power", "Temperature", "Humidity",
    "Global_Horizontal_Irradiance", "Diffuse_Horizontal_Irradiance",
    "Wind_Direction", "Daily_Precipitation",
    "Tilted_Global_Irradiance", "Tilted_Diffuse_Irradiance"
]

correlation_matrix = df[features].corr()

pv_correlation = correlation_matrix["PV_Active_Power"].sort_values(ascending=False)

print("Correlation with PV_Active_Power:\n", pv_correlation)

plt.figure(figsize=(10, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Correlation"}
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)

plt.show()
