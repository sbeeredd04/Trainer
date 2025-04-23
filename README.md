# CIFAR‑100 ResNet‑34 Training Pipeline

This README walks you through every line of code in `cifar100_pipeline.ipynb`, explaining from first principles how and why each step is used. No prior deep‑learning knowledge is assumed—you will learn what an “epoch” is, why we normalize images, what MixUp and CutMix do, and much more.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Environment Setup](#environment-setup)  
3. [Device Selection](#device-selection)  
4. [Hyperparameters Explained](#hyperparameters-explained)  
5. [Data Handling](#data-handling)  
   - [CIFAR‑100 Dataset](#cifar-100-dataset)  
   - [Normalization](#normalization)  
   - [Basic Augmentation](#basic-augmentation)  
   - [Advanced Augmentation (Albumentations)](#advanced-augmentation-albumentations)  
   - [DataLoader](#dataloader)  
6. [Model Architecture: ResNet‑34](#model-architecture-resnet-34)  
7. [Loss Function: Label Smoothing](#loss-function-label-smoothing)  
8. [Optimizer & Learning Rate Scheduler](#optimizer-learning-rate-scheduler)  
9. [Automatic Mixed Precision (AMP)](#automatic-mixed-precision-amp)  
10. [Metrics & Logging](#metrics-logging)  
11. [MixUp & CutMix Regularization](#mixup-cutmix-regularization)  
12. [Training Loop Details](#training-loop-details)  
13. [Evaluation Loop](#evaluation-loop)  
14. [Running the Full Training](#running-the-full-training)  
15. [Usage](#usage)  

---

## Project Overview

We train a **ResNet‑34** model from scratch on the **CIFAR‑100** dataset (60 000 small 32×32 color images in 100 classes). Our goal is 70–75 % top‑1 accuracy. To reach this, we use:

- Strong augmentations (MixUp, CutMix)  
- Label smoothing (softens hard labels)  
- SGD with momentum & weight decay (robust optimizer)  
- CosineAnnealingWarmRestarts LR schedule (dynamic learning rate)  
- Automatic Mixed Precision (faster, uses less GPU memory)  
- TensorBoard logging (track metrics over time)  

All code lives in `resnet34-cifar100/notebooks/cifar100_pipeline.ipynb`. Read on for a line‑by‑line explanation.

---

## Environment Setup

```bash
# Clone the repository and enter the folder
git clone <repo_url>
cd cifar100-resnet

# Create a Python 3.8+ virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required libraries
pip install torch torchvision albumentations torchmetrics tensorboard matplotlib
```

---

## Device Selection

We detect whether a GPU is available; if not, we train on CPU.

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
```

- `cuda`: NVIDIA GPU—much faster for deep learning.  
- `cpu`: your computer’s processor—used if no GPU is found.  

All tensors and the model are moved to this `device`.

---

## Hyperparameters Explained

A **hyperparameter** is a setting you choose before training. You will see these at the top of the notebook:

```python
epochs          = 400      # Number of full passes over the training data
batch_size      = 128      # Number of samples processed at once
lr              = 0.1      # Initial learning rate (step size for optimizer)
momentum        = 0.9      # Momentum factor (smooths gradient updates)
weight_decay    = 1e-3     # L2 penalty to prevent overfitting
label_smoothing = 0.1      # Softens one-hot targets to improve generalization
mixup_alpha     = 1.0      # MixUp interpolation strength
cutmix_alpha    = 1.0      # CutMix patch size control
use_mixup       = True     # Toggle MixUp augmentation
use_cutmix      = True     # Toggle CutMix augmentation
```

- **Epoch**: one complete iteration over the training set.  
- **Batch size**: trade‑off between stable gradient estimates (large batches) and memory constraints.  
- **Learning rate**: how big a step we take in each optimizer update.  

---

## Data Handling

### CIFAR‑100 Dataset

- 50 000 training images, 10 000 test images  
- Each image: 32×32 pixels, 3 color channels (RGB)  
- 100 classes, e.g., “apple,” “tractor,” “orchid,” …

We fetch it in code:

```python
from torchvision import datasets
train_ds = datasets.CIFAR100('./data', train=True,  download=True, transform=...)
test_ds  = datasets.CIFAR100('./data', train=False, download=True, transform=...)
```

### Normalization

Raw pixels range 0–255. We convert them to floats 0–1, then normalize each channel to zero mean and unit variance:

```python
data_mean = [0.5071, 0.4865, 0.4409]
data_std  = [0.2673, 0.2564, 0.2762]
# For each pixel: (value - mean) / std
```

This speeds up and stabilizes training.

### Basic Augmentation

Using `torchvision.transforms`:

```python
from torchvision import transforms as T

torch_transforms = {
  'train': T.Compose([
    T.RandomCrop(32, padding=4),        # Random 32×32 crop with 4‑pixel padding
    T.RandomHorizontalFlip(),           # Flip image left-right 50% of the time
    T.ToTensor(),                       # Convert PIL image to FloatTensor 0–1
    T.Normalize(data_mean, data_std)    # Apply normalization
  ]),
  'test': T.Compose([
    T.ToTensor(),
    T.Normalize(data_mean, data_std)
  ])
}
```

### Advanced Augmentation (Albumentations)

Albumentations provides more varied transforms:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_alb_transforms():
  return A.Compose([
    A.RandomCrop(32,32,p=1.0),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
      A.ColorJitter(...,p=0.5),
      A.ToGray(p=0.2),
      A.CLAHE(p=0.3),
    ], p=0.5),
    A.OneOf([
      A.CoarseDropout(...,p=0.5),
      A.GridDistortion(p=0.2),
      A.ElasticTransform(p=0.3),
    ], p=0.5),
    A.Normalize(mean=data_mean, std=data_std),
    ToTensorV2(),
  ])
```

### DataLoader

We wrap datasets in a DataLoader to:

- **Batch** the data  
- **Shuffle** training data each epoch  
- **Parallelize** data loading with `num_workers`  
- **Pin memory** for faster GPU transfers  

```python
from torch.utils.data import DataLoader
loaders = {
  'train': DataLoader(train_ds, batch_size, shuffle=True,  num_workers=8, pin_memory=True),
  'test' : DataLoader(test_ds,  batch_size, shuffle=False, num_workers=8, pin_memory=True)
}
```

---

## Model Architecture: ResNet‑34

We import and instantiate a 34‑layer Residual Network from `torchvision`:

```python
from torchvision.models import resnet34
model = resnet34(weights=None, num_classes=100).to(device)
```

- **Residual blocks** help gradients flow in deep nets by adding “skip connections.”  
- `weights=None` → train from scratch.  
- `num_classes=100` matches CIFAR‑100.

---

## Loss Function: Label Smoothing

Standard cross‑entropy loss uses hard one‑hot labels (0 or 1). Label smoothing replaces the 1’s with `1 - smoothing` and distributes the remainder over other classes:

```python
import torch.nn as nn
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

This prevents the model from becoming over‑confident and often improves generalization.

---

## Optimizer & Learning Rate Scheduler

### SGD with Momentum & Weight Decay

```python
import torch.optim as optim
optimizer = optim.SGD(
  model.parameters(),
  lr=lr,
  momentum=momentum,
  weight_decay=weight_decay
)
```

- **Momentum** accumulates past gradients to smooth updates.  
- **Weight decay** (L2 regularization) discourages very large weights.

### CosineAnnealingWarmRestarts

We periodically restart the learning rate to escape local minima:

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
```

- `T_0=50`: first cycle length is 50 epochs.  
- `T_mult=2`: each subsequent cycle doubles in length.

---

## Automatic Mixed Precision (AMP)

Using half‑precision (float16) where safe speeds up training and reduces GPU memory:

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

- `autocast()`: runs forward pass automatically in mixed precision.  
- `GradScaler`: scales up loss to avoid underflow in float16.

---

## Metrics & Logging

We track accuracy and write data to TensorBoard:

```python
from torchmetrics.classification import MulticlassAccuracy
accuracy = MulticlassAccuracy(num_classes=100).to(device)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/cifar100_experiment')
```

- **accuracy**: fraction of correct predictions.  
- **SummaryWriter**: logs scalars (loss, accuracy) for real‑time visualization.

---

## MixUp & CutMix Regularization

### MixUp

Interpolates pairs of images and labels:

```python
import numpy as np
import torch

def mixup_data(x, y, alpha):
  lam = np.random.beta(alpha, alpha)
  idx = torch.randperm(x.size(0))
  x2, y2 = x[idx], y[idx]
  mixed_x = lam * x + (1 - lam) * x2
  return mixed_x, y, y2, lam
```

- `lam` sampled from a Beta distribution controlled by `alpha`.  
- Blends two images and their labels proportionally.

### CutMix

Replaces a random patch of one image with a patch from another:

```python
def cutmix_data(x, y, alpha):
  lam = np.random.beta(alpha, alpha)
  b, _, h, w = x.size()
  idx = torch.randperm(b)
  x2, y2 = x[idx], y[idx]
  cx, cy = np.random.randint(w), np.random.randint(h)
  rw, rh = int(w * np.sqrt(1-lam)), int(h * np.sqrt(1-lam))
  x[:, :, cy:cy+rh, cx:cx+rw] = x2[:, :, cy:cy+rh, cx:cx+rw]
  lam = 1 - (rw*rh)/(w*h)
  return x, y, y2, lam
```

- Cuts a rectangular patch out of one image and pastes from another.  
- Adjusts `lam` to reflect the area proportion.

---

## Training Loop Details

```python
def train_epoch(epoch):
  model.train()                         # Enables dropout, batch-norm in “train” mode
  for i, (x, y) in enumerate(loaders['train']):
    x, y = x.to(device), y.to(device)

    # Apply MixUp/CutMix if enabled
    if use_mixup:
      x, y1, y2, lam = mixup_data(x, y, mixup_alpha)
    if use_cutmix:
      x, y1, y2, lam = cutmix_data(x, y, cutmix_alpha)

    optimizer.zero_grad()               # Clear previous gradients
    with autocast():                    # Mixed precision context
      preds = model(x)                  # Forward pass
      if use_mixup or use_cutmix:
        loss = lam*criterion(preds, y1) + (1-lam)*criterion(preds, y2)
      else:
        loss = criterion(preds, y)

    scaler.scale(loss).backward()       # Backpropagate with scaling
    scaler.step(optimizer)              # Optimizer update
    scaler.update()                     # Update the scale factor

    # Compute accuracy on original or mixed labels
    acc = accuracy(preds, y if not (use_mixup or use_cutmix) else y1)

    # Every 100 batches, print a progress message
    if i % 100 == 0:
      print(f"[TRAIN] Ep{epoch+1} B{i} Loss={loss:.4f} Acc={acc:.4f}")

  # Step the learning-rate scheduler once per epoch
  scheduler.step()
```

- **`model.train()`**: sets dropout and batch‑norm layers to training behavior.  
- **`zero_grad()`**: prevents gradient accumulation.  
- **`backward()`**: computes gradients.  
- **`step()`**: updates model weights.  
- **`scheduler.step()`**: updates learning rate.

---

## Evaluation Loop

```python
def evaluate():
  model.eval()                         # “Eval” mode disables dropout, uses running stats
  total_loss = total_acc = 0
  with torch.no_grad():                # No gradient computation needed
    for x, y in loaders['test']:
      x, y = x.to(device), y.to(device)
      preds = model(x)
      total_loss += criterion(preds, y).item()
      total_acc  += accuracy(preds, y).item()
  n = len(loaders['test'])
  return total_loss/n, total_acc/n
```

- We accumulate loss and accuracy over all test batches, then average.

---

## Running the Full Training

```python
for epoch in range(epochs):
  train_epoch(epoch)                           # One pass over training data
  val_loss, val_acc = evaluate()               # Compute on test set
  print(f"[VAL] Ep{epoch+1} Loss={val_loss:.4f} Acc={val_acc:.4f}")

  # Log to TensorBoard
  writer.add_scalar('val_loss', val_loss, epoch)
  writer.add_scalar('val_acc',  val_acc,  epoch)

  # Save a checkpoint every 20 epochs
  if (epoch + 1) % 20 == 0:
    torch.save(model.state_dict(), f"ckpt_ep{epoch+1}.pt")

writer.close()                                 # Finalize TensorBoard log
```

- **Checkpointing**: saves model weights so you can resume or evaluate later.

---

## Usage

- **Jupyter Notebook**: Open `resnet34-cifar100/notebooks/cifar100_pipeline.ipynb` in VS Code or JupyterLab and run cells one by one.  
- **CLI Script**: Wrap the same code into `train_cifar100.py` with `argparse` flags to toggle MixUp/CutMix and adjust hyperparameters.

---

With this detailed guide, you now understand every step—from data loading and augmentation to model training, evaluation, and logging. Happy training!


- try to expereiment with different learning rates
- Hyperparameter tuning / experimenting
- graph of loss 
- graphs for understanding
- take one image run any convolution filter and see how does stride kernel and padding to observe the affect
- 