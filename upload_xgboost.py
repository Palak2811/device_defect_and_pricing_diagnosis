# scripts/upload_xgboost.py
from huggingface_hub import HfApi, create_repo
import shutil
import os

HF_USERNAME = "palakmathur"  # ← CHANGE THIS
repo_id = f"{HF_USERNAME}/device-defect-pricing"

# Create repository for pricing model
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

# Upload XGBoost model file
api = HfApi()
api.upload_file(
    path_or_fileobj="models/price_model.pkl",
    path_in_repo="price_model.pkl",
    repo_id=repo_id,
    repo_type="model"
)

# Upload defect database
api.upload_file(
    path_or_fileobj="data/defect_database.json",
    path_in_repo="defect_database.json",
    repo_id=repo_id,
    repo_type="model"
)

# Upload device catalog
api.upload_file(
    path_or_fileobj="data/device_catalog.json",
    path_in_repo="device_catalog.json",
    repo_id=repo_id,
    repo_type="model"
)

print(f"✅ Pricing model uploaded to: https://huggingface.co/{repo_id}")