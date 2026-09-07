import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms


# -------------------------
# DEVICE
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# IMAGE TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# LOAD FOOD-101
# -------------------------
train_dataset = datasets.Food101(
    root="./data",
    split="train",
    download=False,
    transform=train_transform
)

test_dataset = datasets.Food101(
    root="./data",
    split="test",
    download=False,
    transform=test_transform
)


# -------------------------
# RANDOM SUBSETS
# -------------------------
torch.manual_seed(42)

train_indices = torch.randperm(
    len(train_dataset)
)[:5000]

test_indices = torch.randperm(
    len(test_dataset)
)[:2000]

train_subset = Subset(
    train_dataset,
    train_indices
)

test_subset = Subset(
    test_dataset,
    test_indices
)


# -------------------------
# DATA LOADERS
# -------------------------
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False
)


# -------------------------
# LOAD MOBILENET
# -------------------------
weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)


# -------------------------
# REPLACE CLASSIFIER
# -------------------------
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    101
)


# -------------------------
# LOAD PREVIOUS TRAINED MODEL
# -------------------------
model.load_state_dict(
    torch.load(
        "./models/food101_mobilenet.pth",
        map_location=device
    )
)

print("Previous trained model loaded.")


# -------------------------
# FREEZE MOST FEATURE LAYERS
# -------------------------
for param in model.features.parameters():
    param.requires_grad = False


# Unfreeze last 3 MobileNet feature blocks
for param in model.features[-3:].parameters():
    param.requires_grad = True


# Classifier should remain trainable
for param in model.classifier.parameters():
    param.requires_grad = True


# Move model to CPU/GPU
model = model.to(device)


# -------------------------
# LOSS + OPTIMIZER
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)


# -------------------------
# TRAINING
# -------------------------
epochs = 5

best_validation_accuracy = 0.0

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_accuracy = (
        100 * correct / total
    )

    average_loss = (
        running_loss / len(train_loader)
    )

    print()

    print(
        f"Epoch {epoch + 1} completed"
    )

    print(
        f"Training Loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )


    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    validation_accuracy = (
        100 * correct / total
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )

    print(
        "----------------------------"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            "./models/food101_mobilenet.pth"
        )

        print(
            f"Best model saved! "
            f"Validation Accuracy: "
            f"{best_validation_accuracy:.2f}%"
        )

        print(
            "----------------------------"
        )


print()

print(
    f"Training finished. "
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)
model = models.mobilenet_v3_small(weights=weights)

# 1000 classes -> 101 classes
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    101
)

# Load your already-trained Food-101 weights
model.load_state_dict(
    torch.load(
        "./models/food101_mobilenet.pth",
        map_location=device
    )
)
for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-3:].parameters():
    param.requires_grad = True