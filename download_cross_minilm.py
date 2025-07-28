from sentence_transformers import CrossEncoder

# Define model name and save path
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
save_path = "models/cross_minilm"

# Download and save the model
print(f"Downloading and saving {model_name} to {save_path}...")
model = CrossEncoder(model_name)
model.save(save_path)
print("Cross-encoder download complete.")
