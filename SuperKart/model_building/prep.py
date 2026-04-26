import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/swastisubi/SuperKart/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)

# Cleaning: Dropping unique identifier as it carries no predictive weight
df.drop(columns=['Product_Id'], inplace=True)

# Handling Missing Values (Imputation)
df['Product_Weight'] = df['Product_Weight'].fillna(df['Product_Weight'].mean())
df['Store_Size'] = df['Store_Size'].fillna('Medium')

# Split and Save
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

# Uploading splits to HF for Model Training access
for f in ["train.csv", "test.csv"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id="swastisubi/SuperKart", repo_type="dataset")
