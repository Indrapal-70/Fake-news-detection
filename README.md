# Fake News Detection System

## Problem Statement
Fake news often combines misleading images with deceptive captions. This project builds a multimodal fake news detection system that jointly analyzes image content + textual claims to classify news as Real / Fake.

## System Architecture
```
Image ──► CLIP Image Encoder ──┐
                              ├─► Feature Fusion ─► Classifier ─► Prediction
Text  ──► Text Encoder ───────┘
```

## Tech Stack
- **Image Encoder**: CLIP (ViT-B/32)
- **Text Encoder**: CLIP Text Encoder
- **Fusion**: Concatenation + MLP
- **Classifier**: MLP
- **Framework**: PyTorch
- **UI**: Streamlit

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
