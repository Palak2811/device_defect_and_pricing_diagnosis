# scripts/upload_clip.py
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import create_repo

# Your HF username
HF_USERNAME = "palakmathur"  # ← CHANGE THIS

# Create repository for model
repo_id = f"{HF_USERNAME}/device-defect-clip"

# Create repo
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

print(f"✅ Created repository: https://huggingface.co/{repo_id}")

# Load your fine-tuned model
model = CLIPModel.from_pretrained("models/finetuned_clip/best_model")
processor = CLIPProcessor.from_pretrained("models/finetuned_clip/best_model")

# Push to hub
print("📤 Uploading model... (may take 5-10 minutes)")
model.push_to_hub(repo_id)
processor.push_to_hub(repo_id)

print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")