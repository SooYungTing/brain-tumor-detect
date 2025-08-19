import os, cv2, numpy as np
from collections import Counter
import kagglehub

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
train_dir = os.path.join(path, "Training")
classes = ["glioma", "meningioma", "notumor", "pituitary"]

print("Scanning training folders...")
for cls in classes:
    cls_path = os.path.join(train_dir, cls.lower())
    if not os.path.isdir(cls_path):
        print(f"Missing folder: {cls_path}")
        continue
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    print(f"{cls:10} : {len(imgs):4} images")

print("\nNow loading first 5 images per class to verify pixels...")
images, labels = [], []
for idx, cls in enumerate(classes):
    cls_path = os.path.join(train_dir, cls.lower())
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))][:5]
    for f in imgs:
        img = cv2.imread(os.path.join(cls_path, f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {f}")
            continue
        images.append(img)
        labels.append(idx)

print("Label counts:", Counter(labels))
print("Images loaded:", len(images))