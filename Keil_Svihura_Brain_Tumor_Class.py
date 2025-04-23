# ================================================================
#   RCNN + IKMC (on-line) TENSORFLOW
# ================================================================

# Install required libraries (Colab environment)
#!pip install -q opencv-python-headless scikit-image scikit-learn seaborn

# Import necessary libraries
import os, re, random, cv2, numpy as np, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns
from glob import glob
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras import mixed_precision as mp

# 1 ─── GPU optimization (recommended: Colab Pro/Pro+ with A100)
mp.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# Set seed for reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# 2 ─── Basic constants
TRAIN_DIR = '/content/archive/Training'  # Path to training dataset
TEST_DIR  = '/content/archive/Testing'   # Path to testing dataset
IMG_SIZE  = 224
BATCH     = 16                           # Smaller batch size – online IKMC runs on CPU
EPOCHS    = 30

# 3 ─── Function to extract valid class names (directories with image files)
VALID_EXT = re.compile(r'\.(jpg|jpeg|png|bmp)$', re.IGNORECASE)
def list_classes(path):
    cls = []
    for d in sorted(os.listdir(path)):
        full = os.path.join(path, d)
        if os.path.isdir(full) and any(VALID_EXT.search(f) for f in os.listdir(full)):
            cls.append(d)
    return cls

CLASSES   = list_classes(TRAIN_DIR)
N_CLASSES = len(CLASSES)
print('Classes found:', CLASSES)

# 4 ─── Preprocessing: adaptive denoising + IKMC segmentation (K-means)
def adaptive_filter(img):
    return cv2.fastNlMeansDenoising(img, None, h=10,
                                    templateWindowSize=7, searchWindowSize=21)

def ikmc_segmentation(img, k=3):
    h, w = img.shape
    xv, yv = np.meshgrid(np.arange(w)/w, np.arange(h)/h)
    X = np.column_stack([img.flatten(), xv.flatten(), yv.flatten()])
    X = (X - X.mean(0)) / (X.std(0)+1e-8)
    lbl = KMeans(k, n_init='auto', random_state=SEED).fit_predict(X)
    tumor = int(np.argmax([img.flatten()[lbl==i].mean() for i in range(k)]))
    mask = (lbl==tumor).reshape(h,w).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    return mask

# Full preprocessing function
def preprocess(path_tensor, augment=False):
    path = path_tensor.numpy().decode('utf-8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = adaptive_filter(img)
    mask = ikmc_segmentation(img)
    img = cv2.bitwise_and(img, img, mask=mask).astype('float32') / 255.0
    img = np.expand_dims(img, -1)

    # Optional augmentation
    if augment:
        if random.random() < 0.5:
            img = np.flip(img, 1)
        if random.random() < 0.3:
            M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), random.uniform(-10, 10), 1)
            img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                                 borderMode=cv2.BORDER_REFLECT_101)[..., None]
    return img

# TensorFlow preprocessing wrapper
def tf_preprocess(path, label, augment=False):
    img = tf.py_function(lambda p: preprocess(p, augment), [path], tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img, label

# 5 ─── Load dataset into tf.data pipeline
def build_dataset(root, augment=False):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_paths = glob(os.path.join(root, cls, '*'))
        paths += cls_paths
        labels += [idx] * len(cls_paths)
    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(tf.one_hot(labels, N_CLASSES))
    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(lambda p, l: tf_preprocess(p, l, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(TRAIN_DIR, augment=True)
test_ds  = build_dataset(TEST_DIR,  augment=False)

# 6 ─── Class weight calculation (to compensate imbalance)
train_counter = Counter()
for cls in CLASSES:
    train_counter[cls] = len(glob(os.path.join(TRAIN_DIR, cls, '*')))
tot = sum(train_counter.values())
class_weight = {i: tot / train_counter[cls] for i, cls in enumerate(CLASSES)}
print('Class weights:', class_weight)

# 7 ─── RCNN model using RCL (recurrent conv layer)
class RCL(layers.Layer):
    def __init__(self, f, t=3, **kw):
        super().__init__(**kw)
        self.t = t
        self.c0 = layers.Conv2D(f, 3, padding='same', activation='relu',
                                kernel_initializer='he_normal')
        self.cr = layers.Conv2D(f, 3, padding='same', activation='relu',
                                kernel_initializer='he_normal')
    def call(self, x):
        x = self.c0(x)
        for _ in range(1, self.t):
            x = self.cr(x)
        return x

def build_rcnn(t=3):
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    for f in [32, 64, 128, 256]:
        x = RCL(f, t)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', dtype='float32')(x)
    return models.Model(inp, out, name='RCNN')

model = build_rcnn()
model.summary()

# 8 ─── Training configuration
opt = tf.keras.optimizers.Adam(
    learning_rate=3e-4,
    beta_1=0.9,
    clipnorm=1.0  # Gradient clipping for stability
)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [
    callbacks.ReduceLROnPlateau('val_loss', patience=6, factor=0.3,
                                verbose=1, min_lr=1e-5),
    callbacks.ModelCheckpoint('best_rcnn.keras', save_best_only=True,
                              monitor='val_loss')
]

# Model training
hist = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cb
)

# 9 ─── Evaluation
loss, acc = model.evaluate(test_ds, verbose=0)
print(f'\nTest accuracy: {acc*100:.2f}%')

# Collect predictions
y_true, y_pred = [], []
for x, y in test_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(p, axis=1))

# Classification report and confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('\nClassification report:\n',
      classification_report(y_true, y_pred,
                            labels=sorted(set(y_true)),
                            target_names=[CLASSES[i] for i in sorted(set(y_true))],
                            digits=4))

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion matrix'); plt.ylabel('True'); plt.xlabel('Predicted')
plt.show()

# 10 ─── Training graphs
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy'); plt.legend(['Train', 'Validation'])

plt.subplot(1,2,2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss'); plt.legend(['Train', 'Validation'])
plt.show()


# ======================================================================================================
# Brain Tumor Classification using Simple RCNN in PyTorch
# Includes segmentation and GLCM feature extraction
# ======================================================================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import torchvision

# Importing GLCM functions, compatible with different scikit-image versions
try:
    from skimage.feature.texture import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import graycomatrix, graycoprops

# 1) DATA LOADING & PREPROCESSING
# Load dataset from HuggingFace
hf        = load_dataset("sartajbhuvaji/Brain-Tumor-Classification")
train_hf  = hf["Training"]
test_hf   = hf["Testing"]

# Image transformation: resize to 128x128, convert to grayscale, normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Custom Dataset class for PyTorch
class BrainTumorDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]  # PIL.Image
        if self.transform:
            img = self.transform(img)
        lbl = self.ds[idx]["label"]
        return img, lbl

# Create datasets and data loaders
train_ds = BrainTumorDataset(train_hf, transform)
test_ds  = BrainTumorDataset(test_hf,  transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

# 2) SEGMENTATION using Improved KMeans
from sklearn.cluster import KMeans
from scipy.ndimage import binary_erosion, binary_dilation

def segment_image(img_t, n=3):
    # Apply KMeans to segment image and extract the brightest region (tumor)
    arr = img_t.squeeze().cpu().numpy()
    flat = arr.reshape(-1, 1)
    km = KMeans(n_clusters=n, random_state=0).fit(flat)
    labels = km.labels_.reshape(arr.shape)
    means = [flat[km.labels_ == i].mean() for i in range(n)]
    brightest = np.argmax(means)
    mask = (labels == brightest).astype(np.uint8)

    # Apply morphological operations to refine the mask
    mask = binary_dilation(mask, iterations=2)
    mask = binary_erosion(mask, iterations=1)
    return mask.astype(np.uint8)

# 3) FEATURE EXTRACTION using GLCM (not used in model, for analysis/demo only)
def extract_glcm_features(img_t, mask):
    img = (img_t.squeeze().cpu().numpy()*255).astype(np.uint8)
    region = img*mask
    glcm = graycomatrix(region, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return {
        "energy":       graycoprops(glcm, "energy")[0,0],
        "contrast":     graycoprops(glcm, "contrast")[0,0],
        "homogeneity":  graycoprops(glcm, "homogeneity")[0,0],
        "dissimilarity":graycoprops(glcm, "dissimilarity")[0,0],
    }

# 4) MODEL DEFINITION: Simple RCNN (2 RCL blocks)
class SimpleRCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.rcl1  = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.rcl2  = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1   = nn.Linear(128 * 32 * 32, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.rcl1(x))
        x = self.pool(x)
        x = F.relu(self.rcl2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Move model to GPU
model = SimpleRCNN(num_classes=len(train_hf.features["label"].names)).to("cuda")

# 5) TRAINING LOOP
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
best_acc = 0; patience=5; wait=0

# Training for up to 30 epochs with early stopping
for epoch in range(1,31):
    model.train()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to("cuda"), lbls.to("cuda")
        preds = model(imgs)
        loss = criterion(preds, lbls)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    # Evaluation on test data
    model.eval()
    correct=total=0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to("cuda"), lbls.to("cuda")
            pred = model(imgs).argmax(1)
            correct += (pred==lbls).sum().item()
            total   += lbls.size(0)
    acc = correct/total
    print(f"Epoch {epoch}  Acc={acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        wait = 0
        torch.save(model.state_dict(),"best.pth")
    else:
        wait += 1
    if wait >= patience:
        print("Early stop")
        break

# 6) FINAL EVALUATION
model.load_state_dict(torch.load("best.pth")); model.eval()

from sklearn.metrics import confusion_matrix
all_p = []; all_l = []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to("cuda")
        pr = model(imgs).argmax(1).cpu().tolist()
        all_p += pr
        all_l += lbls.tolist()

cm = confusion_matrix(all_l, all_p, labels=list(range(4)))
accuracy = np.mean([p == l for p, l in zip(all_p, all_l)])
print("Overall Acc:", accuracy)

# Binary classification metrics (Tumor vs No Tumor)
y_true_bin = [0 if y != 3 else 1 for y in all_l]
y_pred_bin = [0 if y != 3 else 1 for y in all_p]
bcm = confusion_matrix(y_true_bin, y_pred_bin)
TN, FP, FN, TP = bcm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy_bin = (TP + TN) / sum(bcm.ravel())

print(f"\n--- Binary Evaluation (Tumor vs Non-Tumor) ---")
print(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")
print(f"Sensitivity = {sensitivity:.2%}")
print(f"Specificity = {specificity:.2%}")
print(f"Accuracy = {accuracy_bin:.2%}")

# Per-class sensitivity and specificity
for i, name in enumerate(train_hf.features["label"].names):
    TP = cm[i,i]
    FN = cm[i,:].sum() - TP
    FP = cm[:,i].sum() - TP
    TN = cm.sum() - TP - FP - FN
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    print(f"{name}  TP={TP}  FP={FP}  TN={TN}  FN={FN}  Sens={sensitivity:.3f}  Spec={specificity:.3f}")

# 7) SINGLE SAMPLE DEMO
# Show original and segmented mask for one random image
idx = random.randint(0, len(train_ds)-1)
img, lbl = train_ds[idx]
mask = segment_image(img)
feats = extract_glcm_features(img, mask)
print("Label:", train_hf.features["label"].names[lbl])
print("Features:", feats)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.imshow(img.squeeze().cpu().numpy(), cmap="gray"); plt.axis("off"); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(mask, cmap="gray"); plt.axis("off"); plt.title("Mask")
plt.show()
