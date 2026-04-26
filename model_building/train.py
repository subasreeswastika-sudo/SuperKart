# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/swastisubi/SuperKart/Xtrain.csv"
Xtest_path = "hf://datasets/swastisubi/SuperKart/Xtest.csv"
ytrain_path = "hf://datasets/swastisubi/SuperKart/ytrain.csv"
ytest_path = "hf://datasets/swastisubi/SuperKart/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# One-hot encode 'Type' and scale numeric features
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'

]
categorical_features = ['Product_Sugar_Content',
                        'Product_Type',
                        'Store_Size',
                        'Store_Location_City_Type',
                        'Store_Type',
                        'Store_Id'
                        ]


# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost regressor model
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 75, 100],
    'xgbregressor__max_depth': [2, 3, 4],
    'xgbregressor__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbregressor__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.4, 0.5, 0.6],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation using a regression scoring metric
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Evaluation:")
print(f"R2 Score: {r2_score(ytrain, y_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(ytrain, y_pred_train):.4f}")

print("\nTest Evaluation:")
print(f"R2 Score: {r2_score(ytest, y_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(ytest, y_pred_test):.4f}")

# Save best model
joblib.dump(best_model, "SuperKart_model_v2.joblib")
joblib.dump(best_model, "SuperKart_model_v1.joblib")

# Upload to Hugging Face
repo_id = "swastisubi/SuperKart"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="SuperKart_model_v1.joblib",
    path_in_repo="SuperKart_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
