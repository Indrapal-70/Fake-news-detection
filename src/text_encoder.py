import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu", model=None, processor=None):
        super().__init__()
        self.device = device
        
        if model:
            self.model = model
        else:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            # Freeze CLIP parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
        if processor:
            self.processor = processor
        else:
            self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        
        # Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features
