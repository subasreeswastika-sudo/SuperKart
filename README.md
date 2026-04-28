# SuperKart Sales Forecast - End-to-End MLOps Pipeline

An automated Machine Learning Operations (MLOps) pipeline built to predict weekly product store sales revenue using historical data, XGBoost, Hugging Face, and GitHub Actions.

---

## 🎯 Business Objective
To build a scalable, automated, and reliable sales forecasting system that helps the business:
- Predict weekly revenue accurately
- Optimize sales operations by region
- Improve supply chain planning and procurement
- Reduce forecast risks and support data-driven decision making

---

## 🛠️ Tech Stack
- **Model**: XGBoost Regressor
- **Orchestration**: Hugging Face (Datasets, Models, Spaces)
- **Frontend**: Streamlit
- **CI/CD**: GitHub Actions
- **Containerization**: Docker

---

## 📁 Project Structure

```bash
SuperKart/
├── .github/
│   └── workflows/
│       └── pipeline.yml                 # CI/CD Pipeline
├── SuperKart/
│   ├── data/                            # Local data copies
│   ├── model_building/
│   │   ├── prep.py                      # Data cleaning & splitting
│   │   └── train.py                     # Model training & HF upload
│   ├── deployment/
│   │   ├── app.py                       # Streamlit Web App
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── hosting/
│       └── hosting.py                   # Push deployment to HF Space
├── SuperKart.csv                        # Raw dataset
├── README.md
└── requirements.txt                     # For GitHub Actions

🚀 Key Features

Automated data ingestion from Hugging Face
Proper data preprocessing and train-test split
Hyperparameter tuning using GridSearchCV
Model training with XGBoost + Scikit-learn Pipeline
Model registration on Hugging Face Model Hub
Interactive Streamlit web application
Fully automated CI/CD pipeline using GitHub Actions
Dockerized deployment on Hugging Face Spaces


📊 Model Performance

R² Score: ~0.91 (Strong predictive performance)
Key Drivers: Product MRP, Store Type, Store Size, and Allocated Display Area


🔗 Important Links

Live Application (Hugging Face Space):
https://huggingface.co/spaces/swastisubi/SuperKart
Hugging Face Dataset:
https://huggingface.co/datasets/swastisubi/SuperKart
Hugging Face Model:
https://huggingface.co/swastisubi/SuperKart


🏗️ MLOps Pipeline Flow

Data Registration → Hugging Face Dataset
Data Preparation & Split
Model Training + Hyperparameter Tuning
Model Evaluation
Model Registration → Hugging Face Model Hub
Deployment → Streamlit App on Hugging Face Space
CI/CD Automation via GitHub Actions


📸 Screenshots
(Add screenshots here after running the pipeline)

GitHub Repository Structure
GitHub Actions Workflow (Successful Run)
Streamlit App Interface with Prediction
Feature Importance Plot


👨‍💻 Developed By
Subasree
MLOps Project | SuperKart Sales Forecasting

Made with ❤️ using Hugging Face + GitHub Actions
