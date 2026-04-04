import os
from huggingface_hub import HfApi, create_repo
from src.config import HF_USERNAME, HF_TOKEN, WEIGHTS_DIR

def push_model_to_hub(local_model_path: str, repo_name: str):
    """Push a .pt weights file (or folder) to HuggingFace Hub."""
    api = HfApi()
    full_repo = f"{HF_USERNAME}/{repo_name}"
    try:
        create_repo(full_repo, token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    if os.path.isdir(local_model_path):
        api.upload_folder(folder_path=local_model_path,
                          repo_id=full_repo, token=HF_TOKEN)
    else:
        api.upload_file(path_or_fileobj=local_model_path,
                        path_in_repo=os.path.basename(local_model_path),
                        repo_id=full_repo, token=HF_TOKEN)
    url = f"https://huggingface.co/{full_repo}"
    print(f"Model pushed to HuggingFace: {url}")
    return url