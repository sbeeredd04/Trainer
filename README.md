## Updated README for ResNet34 CIFAR-100 Training Pipeline

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Environment Setup](#environment-setup)  
3. [Device Selection](#device-selection)  
4. [Hyperparameters Explained](#hyperparameters-explained)  
5. [Dataset Preparation](#dataset-preparation)  
6. [Model Architecture](#model-architecture)  
7. [Training Loop](#training-loop)  
8. [Evaluation and Metrics](#evaluation-and-metrics)  
9. [Visualization](#visualization)  
10. [Usage](#usage)

---

## Project Overview

We train a **ResNet‑34** model from scratch on the **CIFAR‑100** dataset (60,000 small 32×32 color images in 100 classes). The goal is to achieve 70–75% top-1 accuracy. We employ various advanced techniques to optimize performance:

- Strong augmentations (MixUp, CutMix)  
- Label smoothing (softens hard labels)  
- SGD with momentum & weight decay (robust optimizer)  
- CosineAnnealingWarmRestarts LR schedule (dynamic learning rate)  
- Automatic Mixed Precision (faster, uses less GPU memory)  
- TensorBoard logging (track metrics over time)  

All code resides in `resnet34-cifar100.ipynb`. Read on for a line-by-line explanation.

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
pip install torch torchvision albumentations torchmetrics tensorboard matplotlib scikit-learn
```

---

## Device Selection

We detect whether a GPU is available; if not, we train on CPU:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
```

- `cuda`: Use GPU if available for faster computation.  
- `cpu`: Default if no GPU is found.

---

## Hyperparameters Explained

Here are the key hyperparameters for training:

```python
batch_size = 400
epochs = 120
max_lr = 0.01  # Initial learning rate
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```

- **Epoch**: Number of complete passes over the dataset.  
- **Batch Size**: Number of samples processed at once.  
- **Learning Rate**: Step size for optimizer updates.  
- **Weight Decay**: Regularization to prevent overfitting.

---

## Dataset Preparation

### CIFAR-100 Dataset

- 50,000 training images, 10,000 test images.  
- 100 classes, e.g., "apple", "tractor", "orchid".  

```python
train_data = torchvision.datasets.CIFAR100('./', train=True, download=True)
test_data = torchvision.datasets.CIFAR100('./', train=False, download=True)
```

### Normalization

Normalize the dataset using the mean and standard deviation:

```python
mean = [0.5071, 0.4865, 0.4409]
std = [0.2673, 0.2564, 0.2762]
```

### Data Augmentation

- **Training**: Random crop, horizontal flip, color jitter, normalization.
- **Test**: Only normalization.

```python
transform_train = tt.Compose([
    tt.RandomCrop(32, padding=4),
    tt.RandomHorizontalFlip(),
    tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

transform_test = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(mean, std)
])
```

---

## Model Architecture

### ResNet34 Model

- Modified for CIFAR-100: Adjusts the first convolution layer and removes the maxpool layer for better performance on 32x32 images.

```python
class ResNet34Model(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
```

---

## Training Loop

We use the **OneCycleLR** scheduler for training with dynamic learning rates and gradient clipping for stability.

```python
def fit_one_cycle(epochs, max_lr, model, train_loader, test_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            if grad_clip: nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
        result = evaluate(model, test_loader)
```

---

## Evaluation and Metrics

We track performance with standard metrics like F1 score, accuracy, and confusion matrix.

```python
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score

def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)
```

---

## Visualization

We plot the learning curves for loss and accuracy, as well as the classification report for model evaluation.

```python
import matplotlib.pyplot as plt

def plot_learning_curve(history):
    # Extract values from history
    train_losses = [x.get('train_loss', 0) for x in history]
    val_losses = [x['val_loss'] for x in history]
    val_accs = [x['val_acc'] for x in history]
    
    # Plot the learning curve
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()
```

---

## Usage

1. **Run in Jupyter Notebook**: Open `cifar100_pipeline.ipynb` and run cells step-by-step.
2. **Script Mode**: Alternatively, run the model training and evaluation as a script using `main.py` with command line arguments.

---

This guide provides a complete overview of the ResNet34 model training pipeline for CIFAR-100, from dataset handling and augmentation to training, evaluation, and visualization.

# Experimentation and Analysis

## 1. Learning Rate Exploration
- **Experiment**: Train the model with various learning rates (0.1, 0.01, 0.001, 0.0001)
- **Implementation**: Modify the `max_lr` parameter in the main script.
- **Analysis**: Plot validation accuracy curves for each learning rate to find the optimal range.
- **Learning Goal**: Understand how learning rate affects convergence speed and final accuracy.

## 2. Hyperparameter Sensitivity Analysis
- **Experiment**: Systematically vary batch size, weight decay, and gradient clipping.
- **Implementation**: Create a grid search across key hyperparameters.
- **Analysis**: Generate heatmaps showing performance across different parameter combinations.
- **Learning Goal**: Discover which parameters most significantly impact model performance.

## 3. Augmentation Ablation Study
- **Experiment**: Train with different combinations of augmentations enabled/disabled.
- **Implementation**: Create versions of `transform_train` with specific augmentations removed.
- **Analysis**: Compare final accuracy and generalization gap (train-test accuracy difference).
- **Learning Goal**: Quantify the contribution of each augmentation technique.

## 4. Loss Landscape Visualization
- **Experiment**: Plot loss values across different weight configurations.
- **Implementation**: Use techniques like filter normalization to visualize the loss surface.
- **Analysis**: Compare loss landscapes before and after training.
- **Learning Goal**: Visualize optimization challenges in deep networks.

## 5. Convolution Filter Visualization
- **Experiment**: Apply individual convolution filters to sample CIFAR-100 images.
- **Implementation**: Extract filters from the trained model and apply them to sample images.
- **Analysis**: Display side-by-side comparisons of original and filtered images.
- **Learning Goal**: Understand what features different convolutional layers detect.

## 6. Stride and Padding Effects
- **Experiment**: Create a mini-network with various stride and padding configurations.
- **Implementation**: Apply filters with different stride (1, 2, 3) and padding (0, 1, 2) settings.
- **Analysis**: Visualize how spatial dimensions and feature extraction change.
- **Learning Goal**: Develop intuition for how stride and padding affect feature maps.

## 7. Optimization Algorithm Comparison
- **Experiment**: Train identical models with different optimizers (SGD, Adam, RMSprop).
- **Implementation**: Modify the `opt_func` parameter.
- **Analysis**: Compare convergence speeds and final performance.
- **Learning Goal**: Understand optimizer strengths and weaknesses.

## 8. Learning Rate Schedule Comparison
- **Experiment**: Compare OneCycleLR with step decay and cosine annealing.
- **Implementation**: Modify the scheduler implementation in the training loop.
- **Analysis**: Plot learning rates over time and resulting accuracy curves.
- **Learning Goal**: Understand how different learning rate schedules affect training dynamics.
