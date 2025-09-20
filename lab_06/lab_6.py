# -*- coding: utf-8 -*-
"""
Lab 6 — Text Mining with YouTube Data
Course: CSE477 — Data Mining

Implements Tasks A–G:
- TF–IDF (uni- & bi-grams) on comments & captions
- Keyword overlap (Venn) and uniqueness
- Sentiment analysis (VADER)
- Co-occurring pairs from TF–IDF terms (Lab 3 link)
- Temporal split & keyword shift (Lab 4 link)
- Auto-generated 3–5 insights (saved to Lab6_Insights.txt)

INPUTS (edit if needed):
- comments:  try 'outputs/CLEAN/cleaned_comments.csv' or 'cleaned_comments.csv'
- captions:  try 'outputs/CLEAN/cleaned_captions.csv' or 'cleaned_captions.csv'

REQUIREMENTS:
- pandas, numpy, matplotlib, scikit-learn, matplotlib-venn, vaderSentiment

This script will attempt to install any missing packages automatically.
"""

import os
import ast
import sys
import math
import itertools
import warnings
import subprocess
from collections import Counter
from datetime import datetime

# ------------------------- Light auto-install -------------------------
def _safe_pip_install(mod_name, pip_name=None):
    """
    Ensure an importable module exists. If missing, try to pip install it.

    Args:
      mod_name: name used in 'import ...'
      pip_name: package name on PyPI (defaults to mod_name)
    """
    try:
        __import__(mod_name)
        return True
    except Exception:
        try:
            to_install = pip_name or mod_name
            print(f"[setup] Installing {to_install} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", to_install])
            __import__(mod_name)
            return True
        except Exception as e:
            print(f"[warn] Could not install {to_install}: {e}")
            return False

# Correct mappings for tricky packages
_safe_pip_install("pandas")
_safe_pip_install("numpy")
_safe_pip_install("matplotlib")
_safe_pip_install("sklearn", "scikit-learn")
_safe_pip_install("matplotlib_venn", "matplotlib-venn")
_safe_pip_install("vaderSentiment")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VENN_OK = True
try:
    from matplotlib_venn import venn2  # type: ignore
except Exception:
    VENN_OK = False
    print("[warn] matplotlib-venn unavailable; Venn diagram will be skipped.")

from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------- Config -------------------------
COMMENTS_PATHS = [
    "outputs/CLEAN/cleaned_comments.csv",
    "cleaned_comments.csv",
]
CAPTIONS_PATHS = [
    "outputs/CLEAN/cleaned_captions.csv",
    "cleaned_captions.csv",
]

OUTPUT_ROOT = "Lab6"
EXPORT_DIR = os.path.join(OUTPUT_ROOT, "exports")
FIG_DIR = os.path.join(OUTPUT_ROOT, "figures")
REPORT_DIR = os.path.join(OUTPUT_ROOT, "reports")
DPI = 300

TOP_N_TFIDF_UNI = 15   # Part B: top 15 unigrams
TOP_N_VENN = 20        # Part C: top 20 for overlap
TOP_N_TFIDF_BI = 10    # Part D: top 10 bigrams
COOCCUR_TOP_N = 50     # Part F: save top 50 co-occurring pairs

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------- Helpers -------------------------
def find_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    # If none exists, return first and let pandas raise a clear error
    return candidates[0]

def load_cleaned_df(path, tokens_col="cleaned_tokens"):
    df = pd.read_csv(path)
    if tokens_col not in df.columns:
        raise ValueError(
            f"Expected a '{tokens_col}' column in {path}. "
            "Please ensure your cleaned CSV includes tokenized text."
        )

    # Parse tokens if strings like "['word','word2']"
    def ensure_list(x):
        if isinstance(x, list):
            return [str(t) for t in x]
        if pd.isna(x):
            return []
        if isinstance(x, str):
            x = x.strip()
            # try to parse list-like
            try:
                maybe = ast.literal_eval(x)
                if isinstance(maybe, list):
                    return [str(t) for t in maybe]
            except Exception:
                pass
            # fallback: space separated
            return [w for w in x.split() if w]
        return []

    df[tokens_col] = df[tokens_col].apply(ensure_list)
    # Join back into strings for TF–IDF
    df["joined_text"] = df[tokens_col].apply(lambda toks: " ".join(map(str, toks)))
    return df

def tfidf_top_terms(text_series, min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=15):
    """
    Returns a DataFrame with columns: term, score, rank
    Score is mean tf-idf across documents.

    Adaptive version to avoid 'max_df corresponds to < documents than min_df'
    on small datasets:
      - Drops empty rows
      - Lowers min_df to 1 if n_docs is small
      - Raises max_df to 1.0 if needed
      - Retries with safe defaults if necessary
    """
    texts = (
        text_series.fillna("")
        .astype(str)
        .apply(lambda s: s.strip())
    )
    texts = texts[texts.str.len() > 0]
    if texts.empty:
        return pd.DataFrame(columns=["term", "score", "rank"])

    docs = texts.values.tolist()
    n_docs = len(docs)

    min_df_eff = min_df
    max_df_eff = max_df

    if n_docs < 5:
        min_df_eff = 1
        if isinstance(max_df_eff, float) and max_df_eff < 1.0:
            if int(math.floor(max_df_eff * n_docs)) < min_df_eff:
                max_df_eff = 1.0

    if isinstance(max_df_eff, float) and max_df_eff < 1.0:
        min_required = min_df_eff if isinstance(min_df_eff, int) else int(math.ceil(min_df_eff * n_docs))
        if int(math.floor(max_df_eff * n_docs)) < min_required:
            max_df_eff = 1.0

    def _run(_min_df, _max_df):
        vect = TfidfVectorizer(min_df=_min_df, max_df=_max_df, ngram_range=ngram_range)
        X = vect.fit_transform(docs)
        terms = np.array(vect.get_feature_names_out())
        mean_scores = X.mean(axis=0).A1
        order = np.argsort(-mean_scores)[:min(top_n, len(mean_scores))]
        out = pd.DataFrame({"term": terms[order], "score": mean_scores[order]})
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    try:
        return _run(min_df_eff, max_df_eff)
    except ValueError as e:
        try:
            return _run(1, 1.0)
        except Exception:
            print(f"[warn] TF-IDF failed even with fallback (ngram={ngram_range}). Reason: {e}")
            return pd.DataFrame(columns=["term", "score", "rank"])

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"[save] {path}")

def plot_venn(set_a, set_b, labels=("Comments", "Captions"), outpath=None):
    if not VENN_OK:
        print("[info] Venn skipped; matplotlib-venn not available.")
        return
    plt.figure(figsize=(6, 6))
    venn2([set_a, set_b], set_labels=labels)
    plt.title("Top Keywords Overlap (Top 20)")
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=DPI)
        print(f"[save] {outpath}")
    plt.close()

def sentiment_distribution(text_series):
    """
    Uses VADER compound score:
    compound >= 0.05 -> positive
    compound <= -0.05 -> negative
    else neutral
    Returns dict: {'positive': n, 'neutral': n, 'negative': n}, total
    """
    analyzer = SentimentIntensityAnalyzer()
    counts = Counter()
    for t in text_series.fillna(""):
        score = analyzer.polarity_scores(str(t))["compound"]
        if score >= 0.05:
            counts["positive"] += 1
        elif score <= -0.05:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1
    total = sum(counts.values())
    return counts, total

def plot_sentiment_bar(counts, title, outpath):
    labels = ["positive", "neutral", "negative"]
    values = [counts.get(k, 0) for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Count")
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    print(f"[save] {outpath}")
    plt.close()

def cooccurrence_from_top_terms(df, tokens_col, top_terms, top_n_pairs=50):
    """
    Counts co-occurrence of top_terms within each document (set-based within-doc).
    Returns DataFrame with ['term_a','term_b','count'] sorted desc.
    """
    term_set = set(top_terms)
    counter = Counter()
    for toks in df[tokens_col]:
        if not isinstance(toks, list):
            continue
        present = sorted(set([t for t in toks if t in term_set]))
        for a, b in itertools.combinations(present, 2):
            counter[(a, b)] += 1

    if not counter:
        return pd.DataFrame(columns=["term_a", "term_b", "count"])

    pairs, counts = zip(*counter.most_common(top_n_pairs))
    out = pd.DataFrame(pairs, columns=["term_a", "term_b"])
    out["count"] = counts
    return out

def temporal_split_and_compare(df, date_cols=("published_at", "timestamp", "date", "created_at")):
    """
    Attempts a temporal split into two halves (H1 older, H2 newer).
    If no date column is found, uses index-based halves.
    Returns indices for H1, H2.
    """
    date_col_found = None
    for c in date_cols:
        if c in df.columns:
            date_col_found = c
            break

    if date_col_found:
        dcopy = df.copy()

        def parse_dt(x):
            if pd.isna(x):
                return pd.NaT
            try:
                return pd.to_datetime(x)
            except Exception:
                try:
                    return pd.to_datetime(str(x).split()[0], errors="coerce")
                except Exception:
                    return pd.NaT

        dcopy["_dt_"] = dcopy[date_col_found].apply(parse_dt)
        dcopy = dcopy.sort_values("_dt_")
        n = len(dcopy)
        if n == 0:
            return [], []
        cut = n // 2
        idx_h1 = dcopy.index[:cut].tolist()
        idx_h2 = dcopy.index[cut:].tolist()
        return idx_h1, idx_h2
    else:
        n = len(df)
        cut = n // 2
        return list(range(0, cut)), list(range(cut, n))

def keyword_change_between_halves(text_h1, text_h2, **tfidf_kwargs):
    """
    Computes TF–IDF top terms separately for each half, then merges to see changes.
    Returns DataFrame with term, score_h1, rank_h1, score_h2, rank_h2, delta_score.
    """
    tf1 = tfidf_top_terms(text_h1, **tfidf_kwargs)
    tf2 = tfidf_top_terms(text_h2, **tfidf_kwargs)
    merged = pd.merge(tf1, tf2, on="term", how="outer", suffixes=("_h1", "_h2"))
    merged["score_h1"] = merged["score_h1"].fillna(0.0)
    merged["score_h2"] = merged["score_h2"].fillna(0.0)
    merged["rank_h1"] = merged["rank_h1"].fillna(9999).astype(int)
    merged["rank_h2"] = merged["rank_h2"].fillna(9999).astype(int)
    merged["delta_score"] = merged["score_h2"] - merged["score_h1"]
    merged = merged.sort_values("delta_score", ascending=False)
    return merged

# ------------------------- Main Pipeline -------------------------
def main():
    # Part A — Setup & Recall
    comments_path = find_first_existing(COMMENTS_PATHS)
    captions_path = find_first_existing(CAPTIONS_PATHS)

    print(f"[load] comments -> {comments_path}")
    df_comments = load_cleaned_df(comments_path, tokens_col="cleaned_tokens")
    print(f"[load] captions -> {captions_path}")
    df_captions = load_cleaned_df(captions_path, tokens_col="cleaned_tokens")

    # Quick diagnostics (helpful if data is sparse)
    n_comm_docs = (df_comments["joined_text"].str.strip().str.len() > 0).sum()
    n_capt_docs = (df_captions["joined_text"].str.strip().str.len() > 0).sum()
    print(f"[info] #comment docs: {n_comm_docs}")
    print(f"[info] #caption docs: {n_capt_docs}")

    # Part B — TF–IDF Keyword Extraction (unigrams)
    tf_comments_uni = tfidf_top_terms(
        df_comments["joined_text"], min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=TOP_N_TFIDF_UNI
    )
    tf_captions_uni = tfidf_top_terms(
        df_captions["joined_text"], min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=TOP_N_TFIDF_UNI
    )
    save_csv(tf_comments_uni, os.path.join(EXPORT_DIR, "tfidf_keywords_comments.csv"))
    save_csv(tf_captions_uni, os.path.join(EXPORT_DIR, "tfidf_keywords_captions.csv"))

    # Part C — Keyword & Theme Comparison (Top 20 overlap + Venn)
    tf_comments_top20 = tfidf_top_terms(
        df_comments["joined_text"], min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=TOP_N_VENN
    )
    tf_captions_top20 = tfidf_top_terms(
        df_captions["joined_text"], min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=TOP_N_VENN
    )

    comm_set = set(tf_comments_top20["term"].tolist())
    capt_set = set(tf_captions_top20["term"].tolist())
    inter = sorted(comm_set & capt_set)
    comm_only = sorted(comm_set - capt_set)
    capt_only = sorted(capt_set - comm_set)

    venn_csv = pd.DataFrame({
        "intersection": pd.Series(inter),
        "comments_only": pd.Series(comm_only),
        "captions_only": pd.Series(capt_only),
    })
    save_csv(venn_csv, os.path.join(EXPORT_DIR, "keywords_overlap_top20.csv"))

    plot_venn(
        set_a=comm_set,
        set_b=capt_set,
        labels=("Comments (Top 20)", "Captions (Top 20)"),
        outpath=os.path.join(FIG_DIR, "keywords_venn_top20.png")
    )

    # Part D — N-gram Analysis (bigrams)
    tf_comments_bi = tfidf_top_terms(
        df_comments["joined_text"], min_df=2, max_df=0.85, ngram_range=(2, 2), top_n=TOP_N_TFIDF_BI
    )
    tf_captions_bi = tfidf_top_terms(
        df_captions["joined_text"], min_df=2, max_df=0.85, ngram_range=(2, 2), top_n=TOP_N_TFIDF_BI
    )
    save_csv(tf_comments_bi, os.path.join(EXPORT_DIR, "tfidf_bigrams_comments.csv"))
    save_csv(tf_captions_bi, os.path.join(EXPORT_DIR, "tfidf_bigrams_captions.csv"))

    # Part E — Sentiment Analysis (VADER)
    comm_counts, comm_total = sentiment_distribution(df_comments["joined_text"])
    capt_counts, capt_total = sentiment_distribution(df_captions["joined_text"])

    def counts_to_df(counts, total):
        labels = ["positive", "neutral", "negative"]
        rows = [{"category": k, "count": counts.get(k, 0), "percent": (counts.get(k, 0)/total*100 if total else 0.0)} for k in labels]
        return pd.DataFrame(rows)

    save_csv(counts_to_df(comm_counts, comm_total), os.path.join(EXPORT_DIR, "sentiment_comments.csv"))
    save_csv(counts_to_df(capt_counts, capt_total), os.path.join(EXPORT_DIR, "sentiment_captions.csv"))

    plot_sentiment_bar(comm_counts, "Sentiment Distribution — Comments", os.path.join(FIG_DIR, "sentiment_comments.png"))
    plot_sentiment_bar(capt_counts, "Sentiment Distribution — Captions", os.path.join(FIG_DIR, "sentiment_captions.png"))

    # Part F — Link to Past Labs
    # (i) Co-occurring keyword pairs from union of top TF–IDF unigrams (Top 50 union to be safe)
    tf_comments_50 = tfidf_top_terms(df_comments["joined_text"], min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=50)
    tf_captions_50 = tfidf_top_terms(df_captions["joined_text"], min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=50)
    union_top = sorted(set(tf_comments_50["term"]).union(set(tf_captions_50["term"])))

    cooc_comm = cooccurrence_from_top_terms(df_comments, "cleaned_tokens", union_top, top_n_pairs=COOCCUR_TOP_N)
    cooc_capt = cooccurrence_from_top_terms(df_captions, "cleaned_tokens", union_top, top_n_pairs=COOCCUR_TOP_N)
    save_csv(cooc_comm, os.path.join(EXPORT_DIR, "cooccurrence_pairs_comments.csv"))
    save_csv(cooc_capt, os.path.join(EXPORT_DIR, "cooccurrence_pairs_captions.csv"))

    # (ii) Temporal split & keyword prominence changes (if time exists; else index split)
    idx_h1_c, idx_h2_c = temporal_split_and_compare(df_comments)
    idx_h1_cap, idx_h2_cap = temporal_split_and_compare(df_captions)

    comm_changes = keyword_change_between_halves(
        df_comments.loc[idx_h1_c, "joined_text"],
        df_comments.loc[idx_h2_c, "joined_text"],
        min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=50
    )
    capt_changes = keyword_change_between_halves(
        df_captions.loc[idx_h1_cap, "joined_text"],
        df_captions.loc[idx_h2_cap, "joined_text"],
        min_df=2, max_df=0.85, ngram_range=(1, 1), top_n=50
    )
    save_csv(comm_changes, os.path.join(EXPORT_DIR, "temporal_keyword_change_comments.csv"))
    save_csv(capt_changes, os.path.join(EXPORT_DIR, "temporal_keyword_change_captions.csv"))

    # Part G — Insight Statements (3–5)
    insights = []

    # 1) Theme overlap & divergence
    overlap_ratio = (len(inter) / max(1, len(comm_set | capt_set))) * 100
    alignment = "strong" if overlap_ratio >= 40 else "moderate" if overlap_ratio >= 20 else "limited"
    insights.append(
        f"Overlap between comments and captions (Top 20 keywords) is {len(inter)} terms "
        f"({overlap_ratio:.1f}%), indicating {alignment} thematic alignment."
    )

    # 2) Unique themes callout
    top_comm_unique = ", ".join(comm_only[:3]) if comm_only else "—"
    top_capt_unique = ", ".join(capt_only[:3]) if capt_only else "—"
    insights.append(
        f"Comment-only top keywords (sample): {top_comm_unique}; "
        f"Caption-only top keywords (sample): {top_capt_unique}."
    )

    # 3) Sentiment differences
    def _pct(c, total, k): return (c.get(k, 0) / total * 100 if total else 0.0)
    c_pos, c_neg = _pct(comm_counts, comm_total, "positive"), _pct(comm_counts, comm_total, "negative")
    cap_pos, cap_neg = _pct(capt_counts, capt_total, "positive"), _pct(capt_counts, capt_total, "negative")
    insights.append(
        f"Sentiment: comments are {c_pos:.1f}% positive vs {c_neg:.1f}% negative; "
        f"captions are {cap_pos:.1f}% positive vs {cap_neg:.1f}% negative."
    )

    # 4) Bigram highlights
    b_com = tf_comments_bi["term"].tolist()[:3] if not tf_comments_bi.empty else []
    b_cap = tf_captions_bi["term"].tolist()[:3] if not tf_captions_bi.empty else []
    insights.append(
        "Top bigrams — comments: " + (", ".join(b_com) if b_com else "—") +
        "; captions: " + (", ".join(b_cap) if b_cap else "—") + "."
    )

    # 5) Temporal change (keyword rising most in each)
    if not comm_changes.empty:
        c_rise = str(comm_changes.iloc[0]["term"])
        insights.append(f"In comments, '{c_rise}' rose the most from the first half to the second half.")
    if not capt_changes.empty:
        cap_rise = str(capt_changes.iloc[0]["term"])
        insights.append(f"In captions, '{cap_rise}' rose the most from the first half to the second half.")

    # Write insights to file
    insights_path = os.path.join(REPORT_DIR, "Lab6_Insights.txt")
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write("Lab 6 — Insight Statements\n")
        f.write(f"Generated on: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for i, line in enumerate(insights, 1):
            f.write(f"{i}. {line}\n")
    print(f"[save] {insights_path}")

    print("\n[done] Lab 6 pipeline complete.")
    print(f"Exports: {os.path.abspath(EXPORT_DIR)}")
    print(f"Figures: {os.path.abspath(FIG_DIR)}")
    print(f"Reports: {os.path.abspath(REPORT_DIR)}")

if __name__ == "__main__":
    main()
