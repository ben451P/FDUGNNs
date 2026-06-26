import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import os

root = "../../image_dataset/"
save = True

benign = os.path.join(root, "benign")
malignant = os.path.join(root, "malignant")
gen_malignant = os.path.join(root, "generated_malignant")

len_benign = len(os.listdir(benign))
len_malignant = len(os.listdir(malignant))
len_gen_malignant = len(os.listdir(gen_malignant))

groups = ['Benign', 'Malignant']
# We added 5000 generated images into the original dataset
no_aug = [len_benign, len_malignant]
with_aug = [len_benign, len_malignant + len_gen_malignant]

x = np.arange(len(groups))
width = 0.35

bars1 = plt.bar(x - width/2, no_aug, width, label="Without Augmentation")
bars2 = plt.bar(x + width/2, with_aug, width, label="With Augmentation")

plt.ylabel('Instances')
plt.title('Instances Per Category')
plt.xticks(x, groups)
plt.legend()

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.bar_label(bars1, fmt='{:,.0f}')
plt.bar_label(bars2, fmt='{:,.0f}')

plt.tight_layout()
if save:
    plt.savefig("saved_figs/figure2.png",dpi=300)
plt.show()