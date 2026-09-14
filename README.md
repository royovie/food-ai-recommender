# 🍽️ AI Food Image Recipe Recommender

An end-to-end cloud-based AI application that identifies food from an uploaded image and recommends relevant recipes.

The project combines computer vision, REST APIs, containerization, cloud deployment, Infrastructure as Code, and CI to demonstrate a complete machine-learning application lifecycle.

## Overview

A user uploads a food image through a Streamlit interface. A MobileNetV3 model classifies the image into one of 101 Food-101 categories.

The FastAPI backend evaluates the model's confidence and, when the prediction is sufficiently reliable, queries the Spoonacular API for relevant recipes.

### Application Flow

```text
User uploads food image
        ↓
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
MobileNetV3 Model
        ↓
Top-3 Food Predictions
        ↓
Confidence Check
        ↓
Spoonacular API
        ↓
Recipe Recommendations
```

## ☁️ Cloud Architecture

```text
                    GitHub
                      │
               GitHub Actions CI
                      │
                      ↓
               Source Repository

User
 │
 ↓
Azure Container Apps
 │
 ├── Streamlit Frontend
 │        │
 │        ↓
 │   FastAPI Backend
 │        │
 │        ├── MobileNetV3 Model
 │        │
 │        └── Spoonacular API
 │
 └── Container Images
          ↑
 Azure Container Registry

Infrastructure
      ↑
   Terraform
```

## 🧠 Machine Learning

The image classifier uses **MobileNetV3 Small** with transfer learning on the Food-101 dataset.

### Dataset

Food-101 contains:

- 101 food categories
- 1,000 images per category
- 101,000 total images

The full dataset is excluded from Git and Docker images.

### Training

The initial model was trained with a frozen MobileNetV3 feature extractor. The model was subsequently fine-tuned by unfreezing the final feature blocks and applying image augmentation.

Fine-tuning configuration:

| Setting | Value |
|---|---|
| Architecture | MobileNetV3 Small |
| Dataset | Food-101 |
| Training subset | 5,000 images |
| Validation subset | 2,000 images |
| Epochs | 5 |
| Learning rate | 0.0001 |
| Classes | 101 |

### Results

| Metric | Result |
|---|---:|
| Best validation accuracy | 50.10% |
| Test accuracy | 48.70% |
| Macro F1 | 0.48 |
| Weighted F1 | 0.48 |

The model performs particularly well on visually distinctive foods such as edamame and miso soup, while similar-looking dishes remain more challenging.

This represents an important limitation of the current prototype and an opportunity for future model improvement.

## 🎯 Confidence Handling

The system does not automatically trust every model prediction.

The backend evaluates both:

- Top-1 prediction confidence
- Difference between Top-1 and Top-2 confidence

If confidence is below the configured threshold, the result is marked as:

```text
uncertain
```

and recipes are not automatically recommended.

This prevents low-confidence classifications from immediately producing potentially irrelevant recommendations.

## 🥘 Recipe Recommendation

When the model produces a sufficiently confident classification, the backend queries the Spoonacular API.

The application can return recipe information associated with the detected food category.

Users can also manually refine recipe searches when the image prediction does not match what they intended.

## 🖥️ Frontend

The frontend is built with **Streamlit**.

Features include:

- Food image upload
- AI prediction
- Top prediction confidence
- Top-3 classification results
- Recipe recommendations
- Manual recipe refinement
- Persistent results using Streamlit session state

## ⚙️ Backend

The REST API is built using **FastAPI**.

Main endpoints include:

| Endpoint | Purpose |
|---|---|
| `/` | API root |
| `/health` | Health check |
| `/predict` | Image classification and recipe recommendation |
| `/recipes` | Manual recipe search |
| `/docs` | FastAPI Swagger documentation |

## 🐳 Docker

The frontend and backend run as separate containers.

```text
Browser
   ↓
Streamlit :8501
   ↓
FastAPI :8000
   ↓
ML Model + Spoonacular
```

Run locally with:

```bash
docker compose up --build
```

Frontend:

```text
http://localhost:8501
```

Backend:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

## ☁️ Azure Deployment

The application is deployed using Microsoft Azure.

Services used:

- Azure Container Registry (ACR)
- Azure Container Apps
- Azure Resource Groups

The backend and frontend are packaged as Docker images and stored in Azure Container Registry.

Azure Container Apps runs the containers and provides cloud-hosted endpoints for the application.

Sensitive configuration such as the Spoonacular API key is supplied through environment variables/secrets rather than committed to the repository.

## 🏗️ Infrastructure as Code

Terraform is used to define the Azure infrastructure.

The Terraform configuration manages resources such as:

```text
Resource Group
     ↓
Azure Container Registry
     ↓
Container Apps Environment
     ├── Backend
     └── Frontend
```

Example:

```bash
cd terraform

terraform init
terraform validate
terraform plan
terraform apply
```

Terraform state and sensitive `.tfvars` files are excluded from Git.

## 🔄 Continuous Integration

GitHub Actions provides CI for the repository.

On pushes and pull requests, the workflow performs automated validation such as:

```text
Git Push
   ↓
GitHub Actions
   ├── Python validation
   └── Docker build validation
```

Continuous deployment to Azure is currently performed separately because the development Azure tenant restricts Microsoft Entra application registration.

A future version can use GitHub Actions with Azure workload identity federation/OIDC for automated deployment.

## 🛠️ Technology Stack

### Machine Learning
- Python
- PyTorch
- Torchvision
- MobileNetV3
- Food-101

### Backend
- FastAPI
- Uvicorn
- Python

### Frontend
- Streamlit

### External API
- Spoonacular

### DevOps
- Docker
- Docker Compose
- GitHub Actions
- Git

### Cloud
- Microsoft Azure
- Azure Container Registry
- Azure Container Apps

### Infrastructure as Code
- Terraform

## 📁 Project Structure

```text
food-ai-recommender/
│
├── backend/
│   ├── main.py
│   └── recipe_service.py
│
├── frontend/
│   └── app.py
│
├── ml/
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── food101_classes.txt
│
├── tests/
│
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements-backend.txt
├── requirements-frontend.txt
└── README.md
```

Large datasets, trained model artifacts, API keys, virtual environments, and Terraform state are intentionally excluded from source control.

## 🔐 Security

Secrets are not hard-coded into the application.

The project uses:

- `.env` for local development
- Azure Container Apps secrets/environment variables for cloud deployment
- `.gitignore` to prevent secret files from being committed

Never commit the Spoonacular API key or Azure credentials to the repository.

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd food-ai-recommender
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements-backend.txt
pip install -r requirements-frontend.txt
```

### 4. Configure environment variables

Create a `.env` file:

```text
SPOONACULAR_API_KEY=your_api_key
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

### 6. Start the frontend

```bash
streamlit run frontend/app.py
```

Alternatively, run the containerized application:

```bash
docker compose up
```

## ⚠️ Current Limitations

This project is currently a prototype.

Important limitations include:

- Approximately 48.7% test accuracy across 101 classes
- Training and fine-tuning used subsets of Food-101 due to local compute constraints
- Similar-looking foods can be difficult to distinguish
- Recipe quality depends partly on the external Spoonacular API
- Model artifacts are currently managed separately from the Git repository
- Azure CD is not yet automated

## 🔮 Future Improvements

Potential improvements include:

- Train on the complete Food-101 training set
- Experiment with EfficientNet and larger vision architectures
- Improve data augmentation
- Add ingredient-level recognition
- Add model artifact storage/versioning
- Implement automated Azure deployment with GitHub OIDC
- Add comprehensive unit and integration tests
- Add application monitoring
- Add model performance monitoring
- Compare AWS and Azure deployment architectures
- Evaluate scalability, security, latency, and cloud cost

## 🎓 Project Purpose

This project demonstrates how a machine-learning model can be integrated into a cloud-native application rather than existing only as a training notebook.

It combines:

**Machine Learning + API Development + Frontend Development + Docker + Cloud Computing + Infrastructure as Code + CI**

The project also provides a practical foundation for research into cloud-based AI food recognition and recipe recommendation systems.