import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Hugging Face imports
from huggingface_hub import hf_hub_download, HfApi, create_repo

# ==========================
# 1. Load Data from Hugging Face Dataset
# ==========================
DATASET_REPO = "swastisubi/SuperKart"

def load_hf_csv(filename: str):
    """Reliable way to download CSV from HF dataset"""
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        # token=os.getenv("HF_TOKEN")   # Uncomment if your dataset is private
    )
    return pd.read_csv(local_path)

print("Downloading and loading data...")

Xtrain = load_hf_csv("Xtrain.csv")
Xtest = load_hf_csv("Xtest.csv")
ytrain = load_hf_csv("ytrain.csv").iloc[:, 0]   # Convert to 1D Series
ytest = load_hf_csv("ytest.csv").iloc[:, 0]

print(f"Data loaded successfully!")
print(f"Xtrain shape: {Xtrain.shape}, ytrain shape: {ytrain.shape}")
print(f"Xtest shape: {Xtest.shape}, ytest shape: {ytest.shape}")

# ==========================
# 2. Preprocessing
# ==========================
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'
]

categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Store_Id'
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
)

# ==========================
# 3. Model Pipeline + Grid Search
# ==========================
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', xgb_model)          # Explicit name → easier param grid
])

param_grid = {
    'xgb__n_estimators': [50, 75, 100],
    'xgb__max_depth': [2, 3, 4],
    'xgb__colsample_bytree': [0.4, 0.5, 0.6],
    'xgb__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__reg_lambda': [0.4, 0.5, 0.6],
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print("\nStarting Grid Search...")
grid_search.fit(Xtrain, ytrain)

print("\nBest Params:")
print(grid_search.best_params_)

# ==========================
# 4. Evaluation
# ==========================
best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

print("\nTraining Evaluation:")
print(f"R2 Score: {r2_score(ytrain, y_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(ytrain, y_pred_train):.4f}")

print("\nTest Evaluation:")
print(f"R2 Score: {r2_score(ytest, y_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(ytest, y_pred_test):.4f}")

# ==========================
# 5. Save Model
# ==========================
joblib.dump(best_model, "model.joblib")
print("\nModel saved as model.joblib")

# ==========================
# 6. Upload to Hugging Face Model Repo
# ==========================
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "swastisubi/SuperKart"

# Create repo if it doesn't exist
create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
    token=os.getenv("HF_TOKEN")
)

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=repo_id,
    repo_type="model",
    token=os.getenv("HF_TOKEN")
)

print(f"\nModel successfully uploaded to https://huggingface.co/{repo_id}")
