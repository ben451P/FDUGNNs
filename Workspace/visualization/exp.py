

# # Print graph information
# print("Number of nodes (superpixels):", G.number_of_nodes())
# print("Number of edges (adjacency relationships):", G.number_of_edges())

# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# groups = ['Benign', 'Malignant']
# method1 = [32500, 532]
# method2 = [32500, 5532]

# x = np.arange(len(groups))  # the label locations
# width = 0.35  # width of the bars

# # Plot
# plt.bar(x - width/2, method1, width)
# bars2 = plt.bar(x + width/2, method2, width, label='With Augmentation')

# # Labels and Titles
# plt.ylabel('Instances')
# plt.title('Instances Per Category')
# plt.xticks(x)
# plt.legend()

# # Show
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# groups = ['Class 1', 'Class 2']
# method1 = [32500, 2000]  # Original data

# x = np.arange(len(groups))  # the label locations
# width = 0.35  # width of the bars

# # Plot - only method1 (original data)
# plt.bar(x, method1, width)

# # Labels and Titles
# plt.ylabel('Instances')
# plt.title('Instances Per Category')
# plt.xticks(x, groups)

# # Show
# plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd

# # Load into DataFrame
# df = pd.read_csv("../summary3.csv")

# # Set model as index
# df.set_index("model", inplace=True)

# # Transpose so metrics are on x-axis
# df_t = df.T

# # Plot
# plt.figure(figsize=(10, 6))
# for model in df_t.columns:
#     plt.plot(df_t.index, df_t[model], marker='o', label=model)

# plt.title("Unaugmented Dataset Model Performaance")
# plt.xlabel("Metric")
# plt.ylabel("Score")
# plt.ylim(0, 1.05)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

