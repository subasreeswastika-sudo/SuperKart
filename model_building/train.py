%%writefile SuperKart/model_building/train.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
from huggingface_hub import HfApi, create_repo

# Load data from HF
train = pd.read_csv("hf://datasets/swastisubi/SuperKart/train.csv")
test = pd.read_csv("hf://datasets/swastisubi/SuperKart/test.csv")

X_train = train.drop('Product_Store_Sales_Total', axis=1)
y_train = train['Product_Store_Sales_Total']
X_test = test.drop('Product_Store_Sales_Total', axis=1)
y_test = test['Product_Store_Sales_Total']

# Preprocessor
numeric_features = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Establishment_Year']
categorical_features = ['Product_Sugar_Content', 'Product_Type', 'Store_Size',
                        'Store_Location_City_Type', 'Store_Type', 'Store_Id']

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
)

# Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', xgb.XGBRegressor(random_state=42, n_jobs=-1))
])

# Hyperparameter Tuning
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.05, 0.1]
}

print("Starting Grid Search...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluation
y_pred = best_model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Fixed RMSE calculation (compatible with sklearn 1.6+)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"RMSE: {rmse:.2f}")

# Save and Upload Model
joblib.dump(best_model, "model.joblib")

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "swastisubi/SuperKart"

create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Fixed RMSE calculation for sklearn 1.6+"
)

print("✅ Best model successfully registered on Hugging Face Model Hub!")
