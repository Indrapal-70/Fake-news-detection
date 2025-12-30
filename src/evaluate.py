import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import os
import json
from src.fusion_model import FusionClassifier

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Check if batch contains embeddings or raw
            if isinstance(batch, list):
                image_embeds, text_embeds, labels = batch
            else:
                image_embeds = batch['image_embeds']
                text_embeds = batch['text_embeds']
                labels = batch['labels']

            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
            labels = labels.to(device)
            
            outputs = model(image_embeds, text_embeds)
            
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }

if __name__ == "__main__":
    # Example usage
    pass
