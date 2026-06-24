import matplotlib.pyplot as plt
import pandas as pd
import os

# Need to load in manual csv files, then need to change label on plot

save = True

csv_summary1 = "summary13.csv"
csv_summary2 = "summary11_slic.csv"

base_path = "../../relevant_results/"

path = os.path.join(base_path, csv_summary1)

df = pd.read_csv(path)

df.set_index("model", inplace=True)

df_t = df.T

plt.figure(figsize=(10, 6))
for model in df_t.columns:
    plt.plot(df_t.index, df_t[model], marker='o', label=model)

plt.title("Unaugmented Dataset Model Performaance")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
if save:
    plt.savefig("saved_figs/figures_3-5.png",dpi=300)
plt.show()