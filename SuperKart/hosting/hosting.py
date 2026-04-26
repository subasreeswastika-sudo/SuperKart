from huggingface_hub import HfApi
import os

# Initialize the API with your Hugging Face Write Token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the entire deployment folder to your Hugging Face Space
api.upload_folder(
    folder_path="SuperKart/deployment",     # The local folder containing app.py, Dockerfile, etc.
    repo_id="swastisubi/SuperKart", # Your target Hugging Face Space repository
    repo_type="space",                      # Specifies the repository type as a Space
    path_in_repo="",                        # Files will be placed in the root of the Space repo
)
