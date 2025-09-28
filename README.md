# 🎬 Netflix Recommendation System – Ensemble Approach  

This project is inspired by the **legendary Netflix Prize competition (2006)**, where the challenge was to predict user ratings and improve movie recommendations.  

---

## 🔹 What this project does
- Builds a **movie recommendation engine** using collaborative filtering + deep learning  
- Combines **SVD (matrix factorization)**, **KNN similarity**, and an **Autoencoder**  
- Blends predictions with a **Ridge Regression ensemble** to boost accuracy  
- Evaluates using **RMSE** to measure recommendation quality  
- Generates **Top-N personalized movie suggestions**  

---

## 🔹 Dataset
- **MovieLens 100k (demo)** → can be scaled to Netflix Prize dataset  
- Format: `user_id, movie_id, rating`  

---

## 🔹 Results
- Base models (SVD / KNN / Autoencoder): **RMSE ~1.07**  
- Ensemble (stacking): **slightly improved RMSE**  
- Shows how blending multiple models improves personalization  

---

## 🔹 Tech Stack
- **Python** (pandas, numpy, scikit-learn)  
- **Surprise** (SVD, KNNBaseline)  
- **PyTorch** (Autoencoder)  

---

## 🔹 Run the project
```bash
pip install scikit-learn pandas numpy surprise torch
python netflix_like_ensemble.py

🔹 References

Netflix Prize (2006)

MovieLens 100k Dataset
