from sentence_transformers import SentenceTransformer

# Define model name and target path
model_name = "all-MiniLM-L6-v2"
save_path = "models/minilm"

# Load and save the model locally
print(f"Downloading and saving {model_name} to {save_path}...")
model = SentenceTransformer(model_name)
model.save(save_path)
print("Download complete.")
