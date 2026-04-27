# for data manipulation
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ====================== CONFIG ======================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

REPO_ID = "swastisubi/SuperKart"
REPO_TYPE = "dataset"

# Optional: customize commit messages
COMMIT_MSG = "Add train/test splits for modeling pipeline"

# ====================== LOAD DATA ======================
try:
    DATASET_PATH = "https://huggingface.co/datasets/swastisubi/SuperKart/SuperKart.csv"
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Remote load failed: {e}. Falling back to local file.")
    try:
        df = pd.read_csv("SuperKart.csv")
        print("Dataset loaded from local file.")
    except Exception as local_e:
        raise RuntimeError(f"Failed to load dataset from both remote and local: {local_e}")

# Drop unique identifier
if 'Product_Id' in df.columns:
    df.drop(columns=['Product_Id'], inplace=True)

target_col = 'Product_Store_Sales_Total'

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None  # add stratify=y if classification
)

# ====================== SAVE SPLITS (with basic cleaning) ======================
# Optional but recommended to reduce schema issues:
# Convert object columns to string and handle potential mixed types
for col in Xtrain.select_dtypes(include=['object']).columns:
    Xtrain[col] = Xtrain[col].astype(str)
    Xtest[col] = Xtest[col].astype(str)

# Save files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Split files saved locally.")

# ====================== UPLOAD TO HF DATASET ======================
api = HfApi(token=HF_TOKEN)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,          # filename in repo root
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=COMMIT_MSG,
        )
        print(f"Successfully uploaded {file_path}")
    except Exception as upload_err:
        print(f"Failed to upload {file_path}: {upload_err}")
        raise

print("All files uploaded successfully to the dataset repository.")
