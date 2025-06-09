import os

b = len(os.listdir(r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign"))
m = len(os.listdir(r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\malignant"))
gm = len(os.listdir(r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\generated_malignant"))

tm = gm + m

print(1 - tm/(b+tm))