from huggingface_hub import HfApi
import os
#Initialize API with token from GitHub Secrets
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id="swastisubi/SuperKart"
api.upload_folder(
    folder_path=".",     # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
print(f"Deployement to {repo_id} successful!")
