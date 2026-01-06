# Heart Disease Prediction – End-to-End MLOps Pipeline

## 📌 Project Overview
This project implements a **complete end-to-end MLOps pipeline** for predicting the risk of heart disease using the **UCI Heart Disease dataset**.  
The solution demonstrates real-world MLOps practices including **EDA, model development, experiment tracking, CI/CD automation, containerization, deployment, monitoring, and logging**.

---

## 🔗 Code Repository
**GitHub Repository:**  
👉 https://github.com/prasadbishnu09/heart-disease-mlops

---

## ⚙️ Setup & Installation Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/prasadbishnu09/heart-disease-mlops.git
cd heart-disease-mlops

Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Model Training
python src/train.py

5️⃣ Start MLflow UI
mlflow ui
Open: http://127.0.0.1:5000

6️⃣ Run API Locally
uvicorn api.main:app --reload

Swagger UI: http://127.0.0.1:8000/docs

📊 EDA and Modelling Choices
1. Exploratory Data Analysis (EDA)

a. Dataset cleaned and preprocessed
b. Missing values handled
c. Feature distributions visualized
d. Correlation heatmap created
e. Class distribution analysis performed to assess imbalance

2. Modelling Decisions
Two models were trained and compared:
a. Logistic Regression
b. Baseline interpretable model

Requires feature scaling

Random Forest

Non-linear ensemble model

Handles feature interactions well

Evaluation Metrics

Accuracy

ROC-AUC score

Cross-validation

Final Model Selection:
Random Forest performed better and was selected as the final model.

📈 Experiment Tracking Summary (MLflow)

MLflow was integrated to track:

Model parameters

Evaluation metrics

Trained model artifacts

Each model training run is logged as a separate MLflow run, enabling:

Easy comparison of Logistic Regression vs Random Forest

Reproducibility of experiments

Traceability of model performance

MLflow UI was used to:

Compare metrics visually

Inspect artifacts

Identify the best-performing model

🔁 CI/CD Pipeline
CI Pipeline (GitHub Actions)

Implemented using GitHub Actions, triggered on every push.

Pipeline stages:

Install dependencies

Run unit tests using Pytest

Fail build on test or code errors

Generate clear logs for debugging

📄 Workflow file: .github/workflows/ci.yml

✔️ Ensures pipeline fails on errors, satisfying production-readiness requirements.

🐳 Containerization & Deployment
Docker Containerization

FastAPI application packaged using Docker

/predict endpoint exposed

Accepts JSON input and returns:

Prediction

Confidence score

Build Docker Image
docker build -t heart-disease-api .

Run Docker Container
docker run -p 8000:8000 heart-disease-api


Swagger UI:

http://127.0.0.1:8000/docs


📡 Monitoring & Logging
Application Logging

Implemented using Python logging module.

Logged information includes:

Request timestamps

Input payloads

Model predictions

Confidence scores

Logs are visible directly in the Docker container runtime, enabling traceability.

Monitoring

API instrumented using prometheus-fastapi-instrumentator

Metrics endpoint exposed at:

/metrics


These metrics can be scraped by Prometheus and visualized in Grafana.

🏗️ Architecture Diagram (High-Level)
User Request (JSON)
        ↓
     FastAPI API
        ↓
   Input Validation
        ↓
   Feature Scaling
        ↓
   ML Model (Random Forest)
        ↓
 Prediction + Confidence
        ↓
 Logging + Metrics

📸 CI/CD and Deployment Workflow Screenshots

Screenshots included in the repository demonstrate:

GitHub Actions pipeline execution

Passing test results

Docker image build

Running container

Swagger API usage

MLflow experiment comparison

📁 Folder: /screenshots (to be included for submission)

✅ Production Readiness Checklist

✔️ Reproducible setup via requirements.txt
✔️ Automated testing with Pytest
✔️ CI/CD pipeline with failure handling
✔️ Dockerized isolated runtime
✔️ Runtime logging and monitoring
✔️ Experiment traceability via MLflow

🏁 Conclusion

This project successfully demonstrates an industry-grade MLOps workflow from data analysis to production deployment. By combining MLflow for experiment tracking, GitHub Actions for CI/CD, Docker for containerization, and FastAPI for serving, the solution mirrors real-world machine learning systems. The pipeline ensures reproducibility, scalability, observability, and reliability, making it suitable for enterprise-level deployment scenarios.