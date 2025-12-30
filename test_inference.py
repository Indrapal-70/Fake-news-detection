import torch
import pandas as pd
from PIL import Image
import os
import sys
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.image_encoder import ImageEncoder
from src.text_encoder import TextEncoder
from src.fusion_model import FusionClassifier

def test_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Data
    val_csv = os.path.join("data", "val.csv")
    if not os.path.exists(val_csv):
        print("Error: data/val.csv not found.")
        return
        
    df = pd.read_csv(val_csv)
    
    # Pick random sample
    sample = df.sample(1).iloc[0]
    img_path = sample['image_path']
    caption = sample['caption']
    label = sample['label']
    
    print(f"\n--- Test Sample ---")
    print(f"Image: {img_path}")
    print(f"Caption: {caption}")
    print(f"Ground Truth: {'FAKE' if label == 1 else 'REAL'}")
    
    # 2. Load Models
    print("\nLoading models...")
    # Using shared CLIP model for memory efficiency if possible
    img_encoder = ImageEncoder(device=device)
    txt_encoder = TextEncoder(device=device, model=img_encoder.model)
    
    fusion_model = FusionClassifier().to(device)
    model_path = os.path.join("data", "best_model.pth")
    if os.path.exists(model_path):
        fusion_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained fusion model weights.")
    else:
        print("Warning: Trained weights not found. Using random weights.")
    fusion_model.eval()
    
    # 3. Inference
    print("Running inference...")
    image = Image.open(img_path).convert('RGB')
    
    with torch.no_grad():
        # Encode
        img_emb = img_encoder(image).to(device)
        txt_emb = txt_encoder([caption]).to(device)
        
        # Predict
        output = fusion_model(img_emb, txt_emb)
        prob = output.item()
        
    prediction = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    print(f"\n--- Result ---")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    if (prediction == "FAKE" and label == 1) or (prediction == "REAL" and label == 0):
        print("✅ Correct Prediction!")
    else:
        print("❌ Incorrect Prediction.")

    # Model Info
    print("\n--- Model Info ---")
    print("NLP Model: CLIP Text Transformer (ViT-B/32 context)")
    print("Image Model: CLIP Vision Transformer (ViT-B/32)")

if __name__ == "__main__":
    test_inference()
