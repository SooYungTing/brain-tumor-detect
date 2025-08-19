import os
os.environ['KERAS_HOME'] = os.path.join(os.getcwd(), 'keras_cache')

import cv2, numpy as np, tensorflow as tf
import kagglehub, seaborn as sns, matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# download the dataset from kaggle
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
train_dir = os.path.join(path, "Training")
test_dir  = os.path.join(path, "Testing")
classes   = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE  = 224

# Data loading and preprocessing
def load_images(directory):
    images, labels = [], []
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(directory, cls.lower())
        if not os.path.isdir(cls_path): continue
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
        for f in files: 
            img_path = os.path.join(cls_path, f)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            images.append(img)
            labels.append(idx)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

x_train, y_train = load_images(train_dir)
x_test, y_test  = load_images(test_dir)

print("Train:", x_train.shape, np.bincount(y_train))
print("Test :", x_test.shape,  np.bincount(y_test))

# one hot encoding + class weights
y_train_onehot = tf.keras.utils.to_categorical(y_train, len(classes))
y_test_onehot  = tf.keras.utils.to_categorical(y_test,  len(classes))

class_weights = compute_class_weight('balanced',
                                     classes=np.arange(len(classes)),
                                     y=y_train)
class_weights = dict(enumerate(class_weights))

# mild augmentation (prevent over-fitting)
aug = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.03,
    height_shift_range=0.03,
    zoom_range=0.05,
    horizontal_flip=False
)
