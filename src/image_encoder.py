import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        # Process images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Get embeddings
        image_features = self.model.get_image_features(**inputs)
        
        # Normalize embeddings
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features
