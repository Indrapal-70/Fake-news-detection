import torch
from torch.utils.data import DataLoader
from src.dataset import FakeNewsDataset
from src.image_encoder import ImageEncoder
from src.text_encoder import TextEncoder
import numpy as np
import os
from tqdm import tqdm

def cache_features(csv_file, output_dir, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dataset = FakeNewsDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    image_encoder = ImageEncoder(device=device)
    text_encoder = TextEncoder(device=device) # Should share weights ideally if using same CLIP model
    # Note: transformers CLIP separates them but they come from same model usually. 
    # My implementation loads fresh model for each. Memory inefficient?
    # Optimization: Pass the model to TextEncoder.
    text_encoder = TextEncoder(device=device, model=image_encoder.model)
    
    all_image_features = []
    all_text_features = []
    all_labels = []
    
    print(f"Extracting features from {csv_file}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'] # Need to ensure dataset returns PIL images or tensors?
            # My dataset returns PIL images. CLIPProcessor handles them.
            # But DataLoader collate_fn might fail on PIL images if not custom.
            # Default collate fails on PIL.
            
            # Fix: Dataset should probably return raw list for processor in this script, 
            # OR we handle batching manually.
            # Let's handle batching manually or use a custom collate.
            pass

    # Re-thinking: standard DataLoader collate converts to tensor. CLIPProcessor expects PIL or Tensor.
    # If I use `transform=None` in Dataset, it returns PIL. DataLoader will try to stack them and fail.
    
    # Better approach for caching: 
    # Iterate dataset one by one or use a custom collate that returns list of PIL images.
    
    pass

def custom_collate(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    return {'images': images, 'texts': texts, 'labels': labels}

def extract_and_save(csv_file, output_prefix, device):
    dataset = FakeNewsDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    img_enc = ImageEncoder(device=device)
    txt_enc = TextEncoder(device=device, model=img_enc.model) # Shared model
    
    img_feats_list = []
    txt_feats_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['images']
            texts = batch['texts']
            labels = batch['labels']
            
            # Encode
            i_f = img_enc(images).cpu()
            t_f = txt_enc(texts).cpu()
            
            img_feats_list.append(i_f)
            txt_feats_list.append(t_f)
            labels_list.append(labels)
            
    # Concatenate
    all_img = torch.cat(img_feats_list).numpy()
    all_txt = torch.cat(txt_feats_list).numpy()
    all_lbl = torch.cat(labels_list).numpy()
    
    # Save
    np.save(f"{output_prefix}_image_features.npy", all_img)
    np.save(f"{output_prefix}_text_features.npy", all_txt)
    np.save(f"{output_prefix}_labels.npy", all_lbl)
    print(f"Saved features to {output_prefix}*")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    base_dir = r"c:\Users\indra\Fake News Detection system\data"
    
    # Train
    extract_and_save(os.path.join(base_dir, "train.csv"), os.path.join(base_dir, "train"), device)
    
    # Val
    extract_and_save(os.path.join(base_dir, "val.csv"), os.path.join(base_dir, "val"), device)
