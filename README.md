# Transformer-Based NLP: Sentiment Classification with Attention, SHAP, and LIME

## 📌 Project Overview
This project fine‑tunes a **BERT-base-uncased** model on the Amazon Polarity dataset for binary sentiment classification (positive/negative).  
Beyond performance, the work emphasizes **interpretability** using:
- Attention layer analysis (heatmaps from multiple layers/heads)  
- SHAP (Shapley Additive Explanations)  
- LIME (Local Interpretable Model-agnostic Explanations)  

The goal is to combine high accuracy with transparent explanations of model predictions.

---

## ⚙️ Environment Setup
### Hardware
- Linux OS  
- Tesla T4 GPU (15.6 GB VRAM)  
- CUDA 12.8  

### Software
- Python 3.12.13  
- PyTorch 2.10.0+cu128  
- HuggingFace Transformers 5.0.0  
- HuggingFace Datasets 4.0.0  
- SHAP 0.51.0  
- LIME  
- scikit-learn, matplotlib, seaborn, bertviz, pandas, numpy, tqdm, ipywidgets  

Install dependencies:
```bash
pip install -r requirements.txt
