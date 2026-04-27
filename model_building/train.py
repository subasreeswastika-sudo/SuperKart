import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from huggingface_hub import HfApi, create_repo

# Load data
Xtrain = pd.read_csv("https://huggingface.co/datasets/swastisubi/SuperKart/Xtrain.csv")
# ... same for others

ytrain = ytrain.iloc[:, 0]
ytest = ytest.iloc[:, 0]

# Preprocessor (unchanged)
numeric_features = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Establishment_Year']
categorical_features = ['Product_Sugar_Content', 'Product_Type', 'Store_Size',
                        'Store_Location_City_Type', 'Store_Type', 'Store_Id']

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # sparse_output=False is safer with newer sklearn
)

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)  # added n_jobs for speed

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', xgb_model)
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
    verbose=1   # helpful to see progress
)

grid_search.fit(Xtrain, ytrain)

print("Best Params:", grid_search.best_params_)

# Evaluation (unchanged)
y_pred_train = grid_search.predict(Xtrain)
y_pred_test = grid_search.predict(Xtest)

print("\nTraining R2:", r2_score(ytrain, y_pred_train))
print("Training MSE:", mean_squared_error(ytrain, y_pred_train))
print("\nTest R2:", r2_score(ytest, y_pred_test))
print("Test MSE:", mean_squared_error(ytest, y_pred_test))

joblib.dump(grid_search.best_estimator_, "model.joblib")

# Upload
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "swastisubi/SuperKart"

create_repo(repo_id, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=repo_id,
    repo_type="model"
)



