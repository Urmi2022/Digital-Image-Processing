# -*- coding: utf-8 -*-
"""
Lab 5: Clustering YouTube Comments — K-Means & DBSCAN
Works on your Lab 2 outputs: cleaned_comments.csv or cleaned_comments.txt

What this script does:
1) LOAD FIRST: Reads your data, audits shape, and normalizes the text column
2) TF-IDF vectorization (up to 2000 features)
3) K-Means:
   - Elbow method (k=2..7) with inertia plot
   - Run K-Means at chosen k; show top terms per cluster
   - 2D visualization using TruncatedSVD (safe for sparse TF-IDF)
4) DBSCAN:
   - Proper sparse scaling via StandardScaler(with_mean=False)
   - Try eps in [0.3, 0.5, 0.7], print cluster/noise counts
   - Visualize best eps using the SAME 2D projection
5) Prompts: Inline printouts guiding your short written answers

Outputs:
- Saves figures in ./Lab5_Clustering/figures/
- Saves tables in  ./Lab5_Clustering/exports/
"""

# -------------------- LOAD THE CSV/TXT FILES FIRST --------------------
import os, ast
import pandas as pd
import numpy as np

# Default paths (change if needed)
COMMENTS_CSV_PATH = "cleaned_comments.csv"
COMMENTS_TXT_PATH = None  # e.g., "/mnt/data/cleaned_comments.txt" if you only have txt

def _parse_tokens_cell(x):
    """Return a list of tokens from either a list, a list-like string, or a plain string."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        # Try Python list literal first
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        # Fallback: whitespace split
        return [t for t in x.split() if t]
    return []

def load_comments_first():
    """Load the dataset (CSV preferred). Returns (df, used_path)."""
    if COMMENTS_CSV_PATH and os.path.exists(COMMENTS_CSV_PATH):
        df = pd.read_csv(COMMENTS_CSV_PATH)
        used = COMMENTS_CSV_PATH
    elif COMMENTS_TXT_PATH and os.path.exists(COMMENTS_TXT_PATH):
        df = pd.read_table(COMMENTS_TXT_PATH)
        used = COMMENTS_TXT_PATH
    else:
        raise FileNotFoundError(
            "Could not find cleaned comments. "
            "Set COMMENTS_CSV_PATH or COMMENTS_TXT_PATH to your Lab 2 output."
        )
    return df, used

df, DATA_PATH = load_comments_first()

# QUICK AUDIT
print(f"[LOAD] Using: {DATA_PATH}")
print(f"[LOAD] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("Columns:", list(df.columns))

# Normalize to a single text column `__text__`
TEXT_COL = None
if "cleaned_tokens" in df.columns:
    # Convert to list of tokens
    tokens = df["cleaned_tokens"].apply(_parse_tokens_cell)
    df["__text__"] = tokens.apply(lambda toks: " ".join(toks))
    TEXT_COL = "__text__"
elif "cleaned_text" in df.columns:
    df["__text__"] = df["cleaned_text"].astype(str)
    TEXT_COL = "__text__"
else:
    # Last resort: use a likely raw text column if present
    for cand in ["comment_text", "text", "body", "content"]:
        if cand in df.columns:
            df["__text__"] = df[cand].astype(str)
            TEXT_COL = "__text__"
            break

if TEXT_COL is None:
    raise ValueError(
        "No usable text column found. Expected 'cleaned_tokens' or 'cleaned_text'. "
        "Please ensure you completed Lab 2 and exported the proper columns."
    )

# Report shape for Step 1 and prompt
print(f"My dataset contains {df.shape[0]} comments and {df.shape[1]} columns.")
print("\nCRITICAL PROMPT #1 (answer in your report):")
print("- Based on the # of comments above, hypothesize the likely quality of clusters (1–2 sentences).")

# -------------------- IMPORTS FOR ANALYSIS --------------------
from typing import List, Tuple, Dict
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD  # safer than dense PCA for sparse matrices
from sklearn.preprocessing import StandardScaler

# -------------------- PREP & OUTPUT DIRS --------------------
ROOT = "Lab5_Clustering"
FIG = os.path.join(ROOT, "figures")
EXP = os.path.join(ROOT, "exports")
os.makedirs(FIG, exist_ok=True)
os.makedirs(EXP, exist_ok=True)

# -------------------- STEP 2: TF-IDF --------------------
print("\n[TF-IDF] Vectorizing text...")
corpus = df["__text__"].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()
print("[TF-IDF] Shape:", X.shape)

# -------------------- STEP 3A: ELBOW METHOD --------------------
print("\n[K-MEANS] Computing Elbow curve (k=2..7)...")
inertia = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(list(K_range), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG, "elbow_kmeans.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"[K-MEANS] Elbow plot saved -> {os.path.join(FIG, 'elbow_kmeans.png')}")

# -------------------- STEP 3B: RUN K-MEANS --------------------
# Choose k here after inspecting elbow_kmeans.png
optimal_k = 5  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ^^^ Set this to what looks reasonable for YOUR dataset

print(f"\n[K-MEANS] Fitting K-Means with k={optimal_k} ...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X)
df["kmeans_label"] = labels_km

# Top terms per cluster
print("\n[K-MEANS] Top terms per cluster:")
top_terms_per_cluster = []
centers = kmeans.cluster_centers_
for i in range(optimal_k):
    center = centers[i]
    top_idx = center.argsort()[-10:][::-1]
    top_terms = [terms[j] for j in top_idx]
    top_terms_per_cluster.append(top_terms)
    print(f"  Cluster {i}: {', '.join(top_terms)}")

pd.DataFrame({f"cluster_{i}": t for i, t in enumerate(top_terms_per_cluster)}).to_csv(
    os.path.join(EXP, "kmeans_top_terms_per_cluster.csv"), index=False
)

# -------------------- STEP 3C: 2D VISUALIZATION (SAFE FOR SPARSE) --------------------
print("\n[VIS] Computing 2D projection (TruncatedSVD)...")
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels_km, cmap='tab10', alpha=0.6, s=12)
plt.title('K-Means Cluster Visualization (2D SVD)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(FIG, "kmeans_clusters_2d.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"[VIS] Saved -> {os.path.join(FIG, 'kmeans_clusters_2d.png')}")

print("\nCRITICAL PROMPT #2 (answer in your report):")
print("- State your chosen k and justify it from the elbow plot + your data context.")
print("- Identify one GOOD cluster (coherent keywords) and one CONFUSING cluster (mixed/ambiguous), with explanation.")

# -------------------- STEP 4: DBSCAN --------------------
print("\n[DBSCAN] Scaling sparse TF-IDF safely (with_mean=False)...")
X_scaled = StandardScaler(with_mean=False).fit_transform(X)

eps_grid = [0.3, 0.5, 0.7]
results = []
print("[DBSCAN] Trying eps values:", eps_grid)
for eps in eps_grid:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    results.append({"eps": eps, "min_samples": 5, "clusters": n_clusters, "noise_points": n_noise})
    print(f"  eps={eps}: clusters={n_clusters}, noise_points={n_noise}")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(EXP, "dbscan_param_sweep.csv"), index=False)
print(f"[DBSCAN] Sweep results saved -> {os.path.join(EXP, 'dbscan_param_sweep.csv')}")

# Pick best eps after looking at the table
best_eps = results[1]["eps"] if len(results) > 1 else eps_grid[0]  # default pick the middle (0.5)
print(f"[DBSCAN] Using best_eps={best_eps} (change this after inspecting results if needed)")
db_final = DBSCAN(eps=best_eps, min_samples=5)
labels_db = db_final.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels_db, cmap='viridis', alpha=0.6, s=12)
plt.title(f'DBSCAN Visualization (eps={best_eps}, -1 is Noise)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Cluster/Noise')
plt.savefig(os.path.join(FIG, f"dbscan_clusters_2d_eps_{str(best_eps).replace('.','_')}.png"), dpi=300, bbox_inches="tight")
plt.close()
fname = f"dbscan_clusters_2d_eps_{str(best_eps).replace('.', '_')}.png"
print(f"[VIS] Saved -> {os.path.join(FIG, fname)}")


print("\nCRITICAL PROMPT #3 (answer in your report):")
print("- Compare K-Means vs DBSCAN for your data: which was more useful/insightful and why?")
print("- Describe a scenario where DBSCAN would outperform K-Means on comment analysis.")

# -------------------- STEP 5: FINAL REFLECTION --------------------
print("\nCRITICAL PROMPT #4 (answer in your report):")
print("- How did your dataset’s size/topic/style impact each algorithm’s results?")
print("- If advising an airline/marketing team, what SINGLE key lesson would you highlight from this lab?")

# -------------------- OPTIONAL: SAVE LABELS --------------------
df_out = df.copy()
df_out["dbscan_label"] = labels_db
df_out[["__text__", "kmeans_label", "dbscan_label"]].to_csv(
    os.path.join(EXP, "labels_assigned.csv"), index=False
)
print(f"\n[DONE] Outputs written to: {os.path.abspath(ROOT)}")