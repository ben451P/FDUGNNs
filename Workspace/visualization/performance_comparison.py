import pandas as pd
import matplotlib.pyplot as plt
import os

# Need to load in manual csv files, then need to change label on plot

save = True

csv_summary1 = "summary13.csv"
csv_summary2 = "summary11_slic.csv"

base_path = "../../relevant_results/"

p1 = os.path.join(base_path, csv_summary1)
p2 = os.path.join(base_path, csv_summary2)

df1 = pd.read_csv(p1)
df2 = pd.read_csv(p2)

df1.set_index("model", inplace=True)
df2.set_index("model", inplace=True)

df1_t = df1.T.drop("loss")
df2_t = df2.T.drop("loss")

plt.figure(figsize=(12, 6))

for model in df1_t.columns:
    plt.plot(df1_t.index, df1_t[model], marker='o', label=f"{model}")
    plt.plot(df2_t.index, df2_t[model], marker='s', linestyle='--', label=f"{model} (Aug)")

plt.title("Model Performance Comparison", fontsize=18)
plt.xlabel("Metric", fontsize=16)
plt.ylabel("Score", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
if save:
    plt.savefig("saved_figs/performance_comparison_basic.png",dpi=300)
plt.show()