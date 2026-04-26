import pandas as pd
import xgboost as xgb
import joblib, os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from huggingface_hub import HfApi

# Loading Splits
train = pd.read_csv("hf://datasets/swastisubi/SuperKart/train.csv")
test = pd.read_csv("hf://datasets/swastisubi/SuperKart/test.csv")

X_train, y_train = train.drop('Product_Store_Sales_Total', axis=1), train['Product_Store_Sales_Total']
X_test, y_test = test.drop('Product_Store_Sales_Total', axis=1), test['Product_Store_Sales_Total']

# Pipeline Configuration
preprocessor = make_column_transformer(
    (StandardScaler(), ['Product_Weight', 'Product_MRP', 'Store_Establishment_Year']),
    (OneHotEncoder(handle_unknown='ignore'), ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Type'])
)

model = make_pipeline(preprocessor, xgb.XGBRegressor(random_state=42))

# Hyperparameter Tuning
param_grid = {'xgbregressor__n_estimators': [50, 100], 'xgbregressor__max_depth': [3, 5]}
grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)

# Evaluation
best_model = grid.best_estimator_
preds = best_model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, preds)}")

# Registration
joblib.dump(best_model, "superkart_model.joblib")
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(path_or_fileobj="superkart_model.joblib", path_in_repo="model.joblib", repo_id="swastisubi/SuperKart", repo_type="model")
