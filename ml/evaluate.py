import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------
# DEVICE
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# TEST TRANSFORM
# -------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# LOAD TEST DATASET
# -------------------------
test_dataset = datasets.Food101(
    root="./data",
    split="test",
    download=False,
    transform=test_transform
)

class_names = test_dataset.classes


# -------------------------
# USE TEST SUBSET
# -------------------------
torch.manual_seed(42)

test_indices = torch.randperm(
    len(test_dataset)
)[:5000]

test_subset = Subset(
    test_dataset,
    test_indices
)

test_loader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False
)


# -------------------------
# LOAD MODEL
# -------------------------
weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    101
)

model.load_state_dict(
    torch.load(
        "./models/food101_mobilenet.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# -------------------------
# EVALUATE
# -------------------------
all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(test_loader):

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

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Processed batch "
                f"{batch_idx + 1}/{len(test_loader)}"
            )


# -------------------------
# OVERALL ACCURACY
# -------------------------
accuracy = 100 * correct / total

print()
print(
    f"Overall Test Accuracy: "
    f"{accuracy:.2f}%"
)


# -------------------------
# CLASSIFICATION REPORT
# -------------------------
print()
print("Classification Report:")
print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)