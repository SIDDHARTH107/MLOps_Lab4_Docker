# Loan Default Prediction: Dockerized ML Application

> **MLOps Lab 4 Submission** | IE 7374 — Machine Learning Operations  
> **Author:** SIDDHARTH MOHAPATRA  
> **Northeastern University** 

---

## Overview

This project is all about **Docker containerization** of a machine learning application that predicts whether a loan applicant is likely to default. This is the part of my MLOps course (Lab 4). The system consists of:

- **Flask REST API** : Serves predictions from a trained Random Forest model
- **Streamlit Dashboard** : Visual frontend for interacting with the prediction API
- **Docker Compose** : Orchestrates both services in isolated containers

---

## Modifications from Original Lab

The original Docker lab (Week 7) containerizes a simple **Flask weather service** that calls an external weather API. This project makes the following modifications:

| Aspect | Original Lab | This Project |
|--------|-------------|--------------|
| **Use Case** | Weather lookup | Loan default prediction (ML) |
| **Backend** | Flask → external API call | Flask → trained ML model inference |
| **ML Component** | None | Random Forest classifier with preprocessing |
| **Containers** | Single container | Multi-container (API + Dashboard) |
| **Orchestration** | Single `docker run` | Docker Compose with networking |
| **Frontend** | None (API only) | Streamlit interactive dashboard |
| **Dataset** | None | Synthetic credit dataset (1000 records) |
| **Health Checks** | None | Docker health check configured |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose Network                │
│                                                         │
│   ┌─────────────────┐       ┌──────────────────────┐   │
│   │   Flask API      │       │  Streamlit Dashboard  │   │
│   │   (Container 1)  │◄──────│  (Container 2)        │   │
│   │                  │  HTTP │                        │   │
│   │  - Loads model   │       │  - User input form     │   │
│   │  - /predict      │       │  - Calls /predict      │   │
│   │  - /features     │       │  - Shows results       │   │
│   │  Port: 5000      │       │  Port: 8501            │   │
│   └─────────────────┘       └──────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
loan-default-docker/
│
├── model/
│   ├── train_model.py        # Generates data, trains model, saves artifacts
│   ├── model.pkl             # Trained Random Forest model
│   ├── scaler.pkl            # StandardScaler for preprocessing
│   └── feature_names.pkl     # List of expected feature names
│
├── app/
│   ├── app.py                # Flask API with /predict endpoint
│   └── requirements.txt      # Python dependencies for API
│
├── dashboard/
│   ├── dashboard.py          # Streamlit frontend application
│   └── requirements.txt      # Python dependencies for dashboard
│
├── data/
│   └── credit_data.csv       # Generated synthetic dataset
│
├── Dockerfile.api            # Docker image recipe for Flask API
├── Dockerfile.dashboard      # Docker image recipe for Streamlit
├── docker-compose.yml        # Multi-container orchestration
├── requirements.txt          # Dependencies for model training
└── README.md                 # This file
```

---

## How to Run

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.11+ (for model training only)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-default-docker.git
cd loan-default-docker
```

### Step 2: Train the Model
```bash
pip install -r requirements.txt
python model/train_model.py
```
This generates the synthetic dataset and saves the trained model files.

### Step 3: Build and Run with Docker Compose
```bash
docker-compose up --build
```

### Step 4: Access the Application
- **Dashboard:** [http://localhost:8501](http://localhost:8501)
- **API Health Check:** [http://localhost:5000](http://localhost:5000)
- **API Prediction:** Send a POST request to [http://localhost:5000/predict](http://localhost:5000/predict)

### Step 5: Stop the Application
```bash
docker-compose down
```

---

## Testing the API

### Using curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 75000,
    "loan_amount": 15000,
    "credit_score": 680,
    "months_employed": 48,
    "num_credit_lines": 5,
    "interest_rate": 12.5,
    "loan_term": 36,
    "dti_ratio": 0.35
  }'
```

### Expected Response:
```json
{
  "prediction": 0,
  "label": "No Default",
  "confidence": {
    "no_default_probability": 0.92,
    "default_probability": 0.08
  },
  "input_received": { ... }
}
```

---

## Key Docker Concepts Demonstrated

- **Dockerfile**: Defines the build steps for each container image
- **Docker Compose**: Orchestrates multiple containers with shared networking
- **Port Mapping**: Exposes container ports to the host machine
- **Environment Variables**: Configures services without hardcoding values
- **Service Discovery**: Containers communicate using service names (`http://api:5000`)
- **Health Checks**: Automated monitoring of container health
- **Dependency Management**: `depends_on` ensures correct startup order

---

## Model Details

- **Algorithm:** Random Forest Classifier (100 trees, max depth 10)
- **Dataset:** Synthetic credit data — 1000 samples, 8 features
- **Features:** income, loan_amount, credit_score, months_employed, num_credit_lines, interest_rate, loan_term, dti_ratio
- **Target:** Binary (0 = No Default, 1 = Default)
- **Preprocessing:** StandardScaler normalization

---

## Technologies Used

- **Python 3.11**
- **Flask** — REST API framework
- **Streamlit** — Interactive dashboard framework
- **scikit-learn** — Machine learning model training
- **Docker** — Containerization
- **Docker Compose** — Multi-container orchestration
