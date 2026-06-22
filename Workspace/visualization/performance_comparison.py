import pandas as pd
import matplotlib.pyplot as plt


# Load CSVs
df1 = pd.read_csv("../summary13.csv")
df2 = pd.read_csv("../summary11_slic.csv")

# Set model as index
df1.set_index("model", inplace=True)
df2.set_index("model", inplace=True)

# Transpose and remove "loss"
df1_t = df1.T.drop("loss")
df2_t = df2.T.drop("loss")

# Plotting
plt.figure(figsize=(12, 6))

for model in df1_t.columns:
    plt.plot(df1_t.index, df1_t[model], marker='o', label=f"{model}")
    # plt.plot(df1_t.index, df1_t[model], marker='o', label=f"{model} (v1)")
    plt.plot(df2_t.index, df2_t[model], marker='s', linestyle='--', label=f"{model} (Aug)")

# Bigger fonts
plt.title("Model Performance Comparison", fontsize=18)
plt.xlabel("Metric", fontsize=16)
plt.ylabel("Score", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()