import os
import torch
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from collections import Counter
import random

from src.plot_image import plot_first_image
from src.train_resnet import train
from evaluate_metrics import plot_metrics, show_distribution

# --- Config ---
SAMPLE_FRACTION = 1.0  # Full dataset
BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU only")


dataset = ImageFolder(os.path.join(os.getcwd(), "dataset/train"), transform=train_transform)
print(f"Classes Found: {dataset.classes}")
print(f"Total of Images: {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
dataloaders = {
    'train': DataLoader(train_data, batch_size=32, shuffle=True),
    'val': DataLoader(val_data, batch_size=32, shuffle=False)
}

# Load pretrained ResNet-18
print("Loading ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

print(f"Train: {len(train_data)} samples | Validation: {len(val_data)} samples")
train_targets = [dataset.targets[i] for i in train_data.indices]
val_targets = [dataset.targets[i] for i in val_data.indices]
print(f"Train distribution: {dict(zip(dataset.classes, [train_targets.count(i) for i in range(len(dataset.classes))]))}")
print(f"Test distribution:  {dict(zip(dataset.classes, [val_targets.count(i) for i in range(len(dataset.classes))]))}")

plot_first_image(dataset)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
], weight_decay=1e-2)
scaler = torch.GradScaler(device=DEVICE.type) if DEVICE.type == "cuda" else None

torch.backends.cudnn.benchmark = True  # Auto-tune convolutions for fixed input size

model = model.to(DEVICE)

model, history = train(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=30,
    warmup_epochs=3,
    scaler=scaler
)

plot_metrics(history)
show_distribution(model, test_loader, val_data)

os.makedirs('./outputs', exist_ok=True)
torch.save(model.state_dict(), './outputs/model_best.pth')
print("Modelo salvo em ./outputs/model_best.pth")
