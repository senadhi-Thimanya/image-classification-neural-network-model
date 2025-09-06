import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder, upload_file

# Load environment variables
load_dotenv()

# Save your token from environment variable
HfFolder.save_token(os.getenv('HF_TOKEN'))

repo_id = "Raemih/fruit-veg-img-classifier"

# Upload the model file
upload_file(
    path_or_fileobj="Image_classify.keras",   # your local model file
    path_in_repo="Image_classify.keras",      # how it should appear in HF repo
    repo_id=repo_id,
    repo_type="model"
)

print("✅ Model uploaded to Hugging Face Hub!")
