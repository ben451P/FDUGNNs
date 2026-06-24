import matplotlib.pyplot as plt
import numpy as np
import os

root = "../../image_dataset/"
save = True

benign = os.path.join(root, "benign")
malignant = os.path.join(root, "malignant")

len_benign = len(os.listdir(benign))
len_malignant = len(os.listdir(malignant))

groups = ['Benign', 'Malignant']
method1 = [len_benign, len_malignant]

x = np.arange(len(groups))
bar_width = 0.35

plt.bar(x, method1, bar_width)

plt.ylabel('Instances')
plt.title('Instances Per Category')
plt.xticks(x, groups)
if save:
    plt.savefig("saved_figs/bonus_len_comp_figure.png",dpi=300)
plt.show()