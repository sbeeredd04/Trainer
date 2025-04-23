#===========
# Imports and Dependencies
#===========
import pandas as pd
import os
import torch
import time
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, f1_score, recall_score

#===========
# Terminal Styling Utilities
#===========

# ANSI color codes for terminal output styling
class Colors:
    HEADER = '\033[95m'  # Purple
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'  
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # Reset formatting

def print_header(msg):
    """Print a header message with purple formatting"""
    print(f"{Colors.HEADER}{Colors.BOLD}[INFO] {msg}{Colors.END}")
    
def print_step(msg):
    """Print a process step with green formatting"""
    print(f"{Colors.GREEN}[STEP] {msg}{Colors.END}")

def print_result(msg):
    """Print a result with blue formatting"""
    print(f"{Colors.BLUE}[RESULT] {msg}{Colors.END}")
    
def print_warning(msg):
    """Print a warning with yellow formatting"""
    print(f"{Colors.YELLOW}[WARNING] {msg}{Colors.END}")
    
def print_debug(msg):
    """Print debug information with standard formatting"""
    print(f"[DEBUG] {msg}")

#===========
# Hyperparameters Configuration
#===========

# Model training hyperparameters
batch_size = 400
epochs = 120
max_lr = 0.01  # Higher learning rate for ResNet34
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

#===========
# Dataset Preparation
#===========

print_header("Downloading and preparing CIFAR-100 dataset...")
train_data = torchvision.datasets.CIFAR100('./', train=True, download=True)

# Calculate dataset statistics for normalization
# Combine all training images into single array: 50000 x 32 x 32 x 3
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
mean = np.mean(x, axis=(0, 1))/255
std = np.std(x, axis=(0, 1))/255
mean = mean.tolist()
std = std.tolist()
print_result(f"Dataset mean: {mean}, std: {std}")

# Define data augmentation and normalization transforms
# Training data: Random crop, horizontal flip, color jitter, normalization
transform_train = tt.Compose([
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    tt.ToTensor(), 
    tt.Normalize(mean, std, inplace=True)
])

# Test data: Only normalization (no augmentation)
transform_test = tt.Compose([
    tt.ToTensor(), 
    tt.Normalize(mean, std)
])

# Create PyTorch datasets
trainset = torchvision.datasets.CIFAR100(
    "./", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    "./", train=False, download=True, transform=transform_test
)

# Create data loaders with optimization settings
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size*2, shuffle=False, pin_memory=True, num_workers=4
)

print_result(f"Training samples: {len(trainset)}, Test samples: {len(testset)}")

#===========
# Device Configuration
#===========

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# Get device and move data to it
device = get_default_device()
print_result(f"Using device: {Colors.BOLD}{device}{Colors.END}")
trainloader = DeviceDataLoader(trainloader, device)
testloader = DeviceDataLoader(testloader, device)

#===========
# Model Architecture
#===========

# Base model class with common training methods
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        """Perform a training step on a batch of data"""
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        """Perform a validation step on a batch of data"""
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        """Process the outputs of validation steps from entire epoch"""
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        """Print epoch results in a formatted way"""
        print(f"{Colors.BOLD}Epoch [{epoch}]{Colors.END}, "
              f"last_lr: {Colors.YELLOW}{result['lrs'][-1]:.5f}{Colors.END}, "
              f"train_loss: {Colors.RED}{result['train_loss']:.4f}{Colors.END}, "
              f"val_loss: {Colors.RED}{result['val_loss']:.4f}{Colors.END}, "
              f"val_acc: {Colors.GREEN}{result['val_acc']:.4f}{Colors.END}")

# Utility function to calculate accuracy
def accuracy(outputs, labels):
    """Calculate classification accuracy"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Create ResNet34 model adapted for CIFAR-100
class ResNet34Model(ImageClassificationBase):
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Start with standard ResNet34 model
        self.model = models.resnet34(pretrained=False)
        
        # Modify first convolution layer to work with 32x32 images
        # Original ResNet34 first layer: 7x7 conv with stride=2
        # Modified for CIFAR: 3x3 conv with stride=1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove maxpool layer which would reduce feature map size too early for CIFAR
        self.model.maxpool = nn.Identity()
        
        # Change final fully connected layer for 100 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)

# Create model and move to device
model = to_device(ResNet34Model(num_classes=100), device)
print_step("ResNet34 model created and moved to device")

#===========
# Training Utilities
#===========

@torch.no_grad()
def evaluate(model, test_loader):
    """Evaluate model performance on validation/test data"""
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, test_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD,
                  debug_freq=10):  # Parameter to control debugging frequency
    """Train the model using the 1cycle learning rate schedule with batch-level debugging"""
    torch.cuda.empty_cache()
    history = []
    
    # Set up optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        
        # Add batch counter for debugging
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass and calculate loss
            loss = model.training_step(batch)
            train_losses.append(loss)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            current_lr = get_lr(optimizer)
            lrs.append(current_lr)
            sched.step()
            
            # Debug output after each batch (or at specified frequency)
            if batch_idx % debug_freq == 0:
                # Calculate batch accuracy
                with torch.no_grad():
                    images, labels = batch
                    outputs = model(images)
                    batch_acc = accuracy(outputs, labels)
                
                # Print training progress with colored formatting
                print(f"{Colors.BOLD}Epoch {epoch+1}/{epochs}{Colors.END} | "
                      f"Batch {Colors.BLUE}{batch_idx+1}/{total_batches}{Colors.END} | "
                      f"Loss: {Colors.RED}{loss.item():.4f}{Colors.END} | "
                      f"Batch Acc: {Colors.GREEN}{batch_acc.item():.4f}{Colors.END} | "
                      f"LR: {Colors.YELLOW}{current_lr:.6f}{Colors.END}")
        
        # Validation phase
        result = evaluate(model, test_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            print_step(f"Saving model checkpoint at epoch {epoch+1}")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': result['val_acc']
            }
            torch.save(checkpoint, f'resnet34_cifar100_epoch{epoch+1}.pt')
            
    return history

#===========
# Model Training
#===========

# Initial evaluation before training
print_header("Performing initial evaluation...")
history = [evaluate(model, testloader)]
print_result(f"Initial validation loss: {Colors.RED}{history[0]['val_loss']:.4f}{Colors.END}, "
            f"accuracy: {Colors.GREEN}{history[0]['val_acc']:.4f}{Colors.END}")

# Train the model
print_header("\nStarting training...\n")
start_time = time.time()
history += fit_one_cycle(
    epochs, 
    max_lr, 
    model, 
    trainloader, 
    testloader, 
    grad_clip=grad_clip, 
    weight_decay=weight_decay, 
    opt_func=opt_func
)
train_time = time.time() - start_time
print_result(f"\nTraining completed in {Colors.BOLD}{train_time/60:.2f}{Colors.END} minutes")

# Saving the final model
print_step("Saving final model...")
torch.save(model.state_dict(), 'resnet34_cifar100_final.pth')

#===========
# Evaluation and Metrics
#===========

def test_label_predictions(model, device, test_loader):
    """Generate predictions for test data"""
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

print_step("Generating test predictions...")
y_test, y_pred = test_label_predictions(model, device, testloader)

# Calculate various evaluation metrics
print_step("Calculating evaluation metrics...")
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
fs = f1_score(y_test, y_pred, average='weighted')
rs = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print metrics with formatting
print_result(f'Confusion matrix shape: {cm.shape}')
print(f"{Colors.BLUE}Classification Report:{Colors.END}")
print(cr)
print_result(f'F1 score: {Colors.GREEN}{fs:.6f}{Colors.END}')
print_result(f'Recall score: {Colors.GREEN}{rs:.6f}{Colors.END}')
print_result(f'Accuracy score: {Colors.GREEN}{accuracy:.6f}{Colors.END}')

# Save classification report to CSV
report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('resnet34_classification_report.csv', index=True)
print_step("Classification report saved to 'resnet34_classification_report.csv'")

# Calculate training accuracy
print_step("Calculating training accuracy...")
y_train, y_pred_train = test_label_predictions(model, device, trainloader)
train_accuracy = accuracy_score(y_train, y_pred_train)
print_result(f'Train accuracy: {Colors.GREEN}{train_accuracy:.6f}{Colors.END}')

#===========
# Visualization Functions
#===========

# Plot learning curve
def plot_learning_curve(history):
    """Plot the learning curve of training and validation loss/accuracy"""
    # Extract values from history
    train_losses = [x.get('train_loss', 0) for x in history]
    val_losses = [x['val_loss'] for x in history]
    val_accs = [x['val_acc'] for x in history]
    
    # Create subplots
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet34_learning_curves.png')
    plt.show()

# Plot classification metrics
def plot_classification(precision, recall, f1_score):
    """Plot precision, recall, and F1-score for all classes"""
    plt.rcParams['font.size'] = 12
    plt.rc('axes', linewidth=1.75)
    marker_size = 8
    figsize = 6
    plt.figure(figsize=(1.4 * figsize, figsize))
    
    plt.subplot(3, 1, 1)
    plt.plot(precision, 'o', markersize=marker_size)
    plt.ylabel('Precision', fontsize=14)
    plt.xticks([])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(3, 1, 2)
    plt.plot(recall, 'o', markersize=marker_size)
    plt.ylabel('Recall', fontsize=14)
    plt.xticks([])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(3, 1, 3)
    plt.plot(f1_score, 'o', markersize=marker_size)
    plt.ylabel('F1-score', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplots_adjust(hspace=0.001)
    plt.tight_layout()
    plt.savefig("resnet34_classification_metrics.pdf")