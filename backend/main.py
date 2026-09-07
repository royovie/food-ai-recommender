from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import Food101
from backend.recipe_service import get_recipes


app = FastAPI(
    title="Food AI Recipe Recommender",
    version="1.0.0"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Food-101 class names
dataset = Food101(
    root="./data",
    split="test",
    download=False
)

class_names = dataset.classes

# Load model
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def root():
    return {
        "message": "Food AI Recipe Recommender API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/recipes")
def recipes(query: str):
    return {
        "query": query,
        "recipes": get_recipes(query)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            3,
            dim=1
        )

        top_predictions = []

        for probability, index in zip(
            top_probs[0],
            top_indices[0]
        ):
            top_predictions.append({
                "food": class_names[index.item()],
                "confidence": round(
                    probability.item() * 100,
                    2
                )
            })

    predicted_class = top_predictions[0]["food"]

    top1_confidence = top_predictions[0]["confidence"]
    top2_confidence = top_predictions[1]["confidence"]

    confidence_gap = top1_confidence - top2_confidence

    if top1_confidence < 50 or confidence_gap < 15:
        predicted_class = "uncertain"
        recipe_results = []
    else:
        recipe_results = get_recipes(predicted_class)

    return {
        "prediction": predicted_class,
        "confidence": top1_confidence,
        "confidence_gap": round(confidence_gap, 2),
        "top_predictions": top_predictions,
        "recipes": recipe_results
    }