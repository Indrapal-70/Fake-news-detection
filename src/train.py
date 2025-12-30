import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np
from src.fusion_model import FusionClassifier
from src.evaluate import evaluate

def load_cached_data(prefix):
    img = np.load(f"{prefix}_image_features.npy")
    text = np.load(f"{prefix}_text_features.npy")
    labels = np.load(f"{prefix}_labels.npy")
    return img, text, labels

def train_model(data_dir, epochs=10, batch_size=32, lr=1e-4, device='cuda'):
    # Load data
    print("Loading cached features...")
    train_img, train_txt, train_lbl = load_cached_data(os.path.join(data_dir, "train"))
    val_img, val_txt, val_lbl = load_cached_data(os.path.join(data_dir, "val"))
    
    # Create Datasets
    train_dataset = TensorDataset(torch.tensor(train_img), torch.tensor(train_txt), torch.tensor(train_lbl).float().unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(val_img), torch.tensor(val_txt), torch.tensor(val_lbl).float().unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = FusionClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for img_emb, txt_emb, labels in loop:
            img_emb, txt_emb, labels = img_emb.to(device), txt_emb.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_emb, txt_emb)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/len(train_loader))
            
        # Validation
        print(f"\nValidation Epoch {epoch+1}:")
        metrics = evaluate(model, val_loader, device)
        
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(data_dir, "best_model.pth"))
            print("Saved Best Model!")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = r"c:\Users\indra\Fake News Detection system\data"
    train_model(data_dir, device=device)
