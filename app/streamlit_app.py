import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
import numpy as np
from src.image_encoder import ImageEncoder
from src.text_encoder import TextEncoder
from src.fusion_model import FusionClassifier

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

st.title("🕵️ Multimodal Fake News Detection")
st.markdown("Upload an image and enter a caption to check if the news is **Real** or **Fake**.")

@st.cache_resource
def load_models():
    device = 'cpu' # Use CPU for inference on local machine usually safer for demo
    
    # Load encoders
    image_encoder = ImageEncoder(device=device)
    text_encoder = TextEncoder(device=device, model=image_encoder.model)
    
    # Load Fusion Model
    fusion_model = FusionClassifier().to(device)
    
    # Load weights
    model_path = os.path.join("data", "best_model.pth")
    if os.path.exists(model_path):
        fusion_model.load_state_dict(torch.load(model_path, map_location=device))
        st.success("Loaded trained model!")
    else:
        st.warning("No trained model found. Using random weights.")
        
    fusion_model.eval()
    return image_encoder, text_encoder, fusion_model, device

image_encoder, text_encoder, fusion_model, device = load_models()

col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    caption = st.text_area("Enter the caption/title:", height=150)
    
    predict_btn = st.button("Detect Fake News", type="primary")

with col2:
    st.header("Analysis")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if predict_btn and uploaded_file and caption:
        with st.spinner("Analyzing..."):
            # Encode
            inputs = [image]
            img_embeds = image_encoder(inputs).to(device)
            
            txt_inputs = [caption]
            txt_embeds = text_encoder(txt_inputs).to(device)
            
            # Predict
            with torch.no_grad():
                output = fusion_model(img_embeds, txt_embeds)
                prob = output.item()
                
            # Display Result
            if prob > 0.5:
                st.error(f"🚨 **FAKE NEWS DETECTED**")
                st.metric("Confidence (Fake)", f"{prob*100:.2f}%")
            else:
                st.success(f"✅ **REAL NEWS**")
                st.metric("Confidence (Real)", f"{(1-prob)*100:.2f}%")
                
            st.progress(prob)
