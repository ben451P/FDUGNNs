import os, random


def load_dataset(dataset_size=None):
    dataset_size = 50
    random.seed(0)

    root = r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign"
    benign_pathnames = [(root + r"""\ """).strip() + file for file in os.listdir(root)]
    benign_pathnames = random.sample(
        benign_pathnames, dataset_size if dataset_size else len(benign_pathnames)
    )

    root = r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\malignant"
    malignant_pathnames = [
        (root + r"""\ """).strip() + file for file in os.listdir(root)
    ]
    malignant_pathnames = random.sample(
        malignant_pathnames, dataset_size if dataset_size else len(malignant_pathnames)
    )

    all_pathnames = benign_pathnames + malignant_pathnames
    labels = [0] * len(benign_pathnames) + [1] * len(malignant_pathnames)

    return all_pathnames, labels
