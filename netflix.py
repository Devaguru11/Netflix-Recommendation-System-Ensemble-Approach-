# netflix_like_ensemble.py
# Run: python netflix_like_ensemble.py
# Requires: pip install scikit-learn pandas numpy surprise torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD, KNNBaseline
from surprise.model_selection import train_test_split as surprise_train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -------------------------
# 1) Load dataset
# -------------------------
# Using MovieLens 100k (built-in) for demo. To use your own CSV later, see comment below.
data = Dataset.load_builtin('ml-100k')  # small demo dataset

# Surprise's train/test split (provides Surprise Dataset objects)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

# Convert testset (list of (uid, iid, r) tuples) to DataFrame for stacking
test_df = pd.DataFrame(testset, columns=['user_raw', 'item_raw', 'rating'])

# -------------------------
# 2) Model A: Matrix Factorization (SVD)
# -------------------------
print("Training SVD (matrix factorization)...")
svd = SVD(n_factors=100, n_epochs=30, biased=True, lr_all=0.005, reg_all=0.02, random_state=RANDOM_SEED)
svd.fit(trainset)

# Helper to get SVD predictions for a list of (u,i) pairs
def svd_predict_many(model, pairs):
    preds = [model.predict(u, i).est for (u, i) in pairs]
    return np.array(preds)

# -------------------------
# 3) Model B: Neighborhood (Item-item)
# -------------------------
print("Training KNNBaseline (neighborhood)...")
sim_options = {'name': 'cosine', 'user_based': False}  # item-item
knn = KNNBaseline(k=40, sim_options=sim_options)
knn.fit(trainset)

def knn_predict_many(model, pairs):
    preds = []
    for (u, i) in pairs:
        try:
            est = model.predict(u, i).est
        except:
            est = trainset.global_mean  # fallback
        preds.append(est)
    return np.array(preds)

# -------------------------
# 4) Model C: Simple Autoencoder CF (PyTorch)
#    - Builds user x item rating matrix and trains a denoising autoencoder
# -------------------------
print("Preparing data for autoencoder...")

# Build full pandas DataFrame from Surprise trainset
# We'll extract user/item raw ids mapping from Surprise trainset
uid_to_inner = {trainset.to_raw_uid(i): i for i in range(trainset.n_users)}  # not used directly
# Build mapping and rating matrix
# Get all unique raw ids from trainset
u_raw_ids = [trainset.to_raw_uid(i) for i in range(trainset.n_users)]
i_raw_ids = [trainset.to_raw_iid(i) for i in range(trainset.n_items)]
u2idx = {u: idx for idx, u in enumerate(u_raw_ids)}
i2idx = {i: idx for idx, i in enumerate(i_raw_ids)}

# Initialize matrix with zeros (unrated=0)
rating_matrix = np.zeros((len(u_raw_ids), len(i_raw_ids)), dtype=np.float32)

# Fill from trainset: iterate over inner ids
for u_inner in range(trainset.n_users):
    u_raw = trainset.to_raw_uid(u_inner)
    for i_inner, rating in trainset.ur[u_inner]:
        i_raw = trainset.to_raw_iid(i_inner)
        rating_matrix[u2idx[u_raw], i2idx[i_raw]] = rating

# Normalize ratings to [0,1] for training stability (original range 1-5)
min_r, max_r = 1.0, 5.0
rating_matrix_norm = (rating_matrix - min_r) / (max_r - min_r)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rating_tensor = torch.tensor(rating_matrix_norm, device=device)

class AutoEncoder(nn.Module):
    def __init__(self, num_items, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items)  # reconstruct all items
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

num_users, num_items = rating_tensor.shape
ae = AutoEncoder(num_items=num_items, hidden_dim=512, latent_dim=128).to(device)
optimizer = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

print("Training autoencoder (this may take a few minutes on CPU)...")
ae.train()
EPOCHS = 30
batch_size = 64
idxs = np.arange(num_users)
for epoch in range(EPOCHS):
    np.random.shuffle(idxs)
    epoch_loss = 0.0
    for st in range(0, num_users, batch_size):
        batch_idx = idxs[st:st+batch_size]
        batch = rating_tensor[batch_idx]  # shape (b, num_items)

        # Optionally add input noise (denoising)
        noisy = batch.clone()
        mask = (torch.rand_like(noisy) < 0.2).float()
        noisy = noisy * (1 - mask)  # drop 20% inputs

        preds = ae(noisy)
        loss = criterion(preds * (batch > 0).float(), batch)  # compute loss only on rated items
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.shape[0]
    epoch_loss /= num_users
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, loss={epoch_loss:.6f}")

# Function to get autoencoder prediction for user u_raw and item i_raw
ae.eval()
with torch.no_grad():
    reconstructed = ae(rating_tensor).cpu().numpy()  # shape (num_users, num_items)

def ae_predict_many(pairs):
    preds = []
    for (u_raw, i_raw) in pairs:
        # Some users/items in test may not be in training set; handle fallback
        if u_raw in u2idx and i_raw in i2idx:
            uidx = u2idx[u_raw]
            iidx = i2idx[i_raw]
            val_norm = reconstructed[uidx, iidx]
            val = val_norm * (max_r - min_r) + min_r
        else:
            val = np.mean(rating_matrix[rating_matrix > 0]) if np.any(rating_matrix > 0) else 3.0
        preds.append(val)
    return np.array(preds)

# -------------------------
# 5) Build stacking dataset
# -------------------------
print("Generating model predictions for stacking...")

pairs = list(zip(test_df['user_raw'], test_df['item_raw']))
y_true = test_df['rating'].values.astype(np.float32)

pred_svd = svd_predict_many(svd, pairs)
pred_knn = knn_predict_many(knn, pairs)
pred_ae = ae_predict_many(pairs)

# Clip predictions to rating range
pred_svd = np.clip(pred_svd, min_r, max_r)
pred_knn = np.clip(pred_knn, min_r, max_r)
pred_ae = np.clip(pred_ae, min_r, max_r)

stack_X = np.vstack([pred_svd, pred_knn, pred_ae]).T
stack_y = y_true

# Evaluate individual base models
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

print("Base model RMSEs on test set:")
print(f" - SVD RMSE: {rmse(pred_svd, y_true):.4f}")
print(f" - KNN RMSE: {rmse(pred_knn, y_true):.4f}")
print(f" - AutoEncoder RMSE: {rmse(pred_ae, y_true):.4f}")

# -------------------------
# 6) Meta-learner (blending)
# -------------------------
print("Training Ridge meta-learner to blend models...")
meta = Ridge(alpha=1.0)
meta.fit(stack_X, stack_y)
stack_pred = meta.predict(stack_X)
print(f" - Blended RMSE: {rmse(stack_pred, y_true):.4f}")

# Show weights for interpretability
print("Meta-learner weights:", meta.coef_, "intercept:", meta.intercept_)

# -------------------------
# 7) Example: predict for a single user-item pair using ensemble
# -------------------------
def ensemble_predict(user_raw, item_raw):
    s = svd.predict(user_raw, item_raw).est
    k = knn.predict(user_raw, item_raw).est
    a = ae_predict_many([(user_raw, item_raw)])[0]
    arr = np.array([s, k, a]).reshape(1, -1)
    return meta.predict(arr)[0]

# Demo predict
sample_row = test_df.sample(1, random_state=RANDOM_SEED).iloc[0]
print("Sample true rating:", sample_row['rating'])
print("SVD:", svd.predict(sample_row['user_raw'], sample_row['item_raw']).est)
print("KNN:", knn.predict(sample_row['user_raw'], sample_row['item_raw']).est)
print("AE:", ae_predict_many([(sample_row['user_raw'], sample_row['item_raw'])])[0])
print("Blended:", ensemble_predict(sample_row['user_raw'], sample_row['item_raw']))

# -------------------------
# 8) How to adapt to Netflix Prize dataset
# -------------------------
adapt_instructions = """
If you'd like to run on the actual Netflix Prize dataset (or any custom ratings CSV):
- Ensure you have CSV with columns: user_id, item_id, rating (and optionally timestamp).
- Load into pandas, then build a Surprise Dataset via:
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[['user_id','item_id','rating']], reader)
- Replace Surprise train/test split above with surprise_train_test_split(data, ...)
- For the autoencoder:
    - Build u_raw_ids = sorted unique user ids from training set
    - Build i_raw_ids = sorted unique item ids from training set
    - Create rating_matrix with shape (n_users, n_items) and fill with training ratings
- Consider sampling or using minibatches for very large datasets, and use GPU for training AE.
- For production deployment, the winning Netflix solutions used many more specialized models and careful blending. This example demonstrates the approach and a simple stacking pipeline.
"""
print(adapt_instructions)

# Save model predictions and stacking results to CSV for slides/report
out_df = pd.DataFrame({
    'user': test_df['user_raw'],
    'item': test_df['item_raw'],
    'true': y_true,
    'svd': pred_svd,
    'knn': pred_knn,
    'ae': pred_ae,
    'blend': stack_pred
})
out_path = "ensemble_predictions_sample.csv"
out_df.to_csv(out_path, index=False)
print(f"Saved example predictions to {out_path}")
