# -*- coding: utf-8 -*-
"""
Incremental Pattern Mining — Load CSVs first, then run chunked analysis.

This script:
1) Loads and audits your Lab 2 outputs (cleaned_comments.csv and optionally cleaned_captions.csv)
2) Runs five-chunk incremental mining on cleaned_comments.csv:
   - top unigrams & co-occurring pairs per chunk
   - tracks frequencies across chunks (line plots)
   - computes correlation matrices for selected patterns
Outputs are written to ./Lab3_Incremental_PatternMining/
"""

# ---------- Load the CSV files first (quick audit) ----------
import os, ast
import pandas as pd
import numpy as np

COMMENTS_PATH = "cleaned_comments.csv"
CAPTIONS_PATH = "cleaned_captions.csv"   # optional, not used in incremental analysis

def _parse_tokens_quick(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except Exception:
            return [t for t in x.split() if t]
    return []

def _audit(path):
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    df = pd.read_csv(path)
    has_tokens = "cleaned_tokens" in df.columns
    nonempty = 0
    tmin = tmax = tavg = None
    if has_tokens:
        toks = df["cleaned_tokens"].apply(_parse_tokens_quick)
        lens = toks.apply(len)
        nonempty = int((lens > 0).sum())
        if len(lens) > 0:
            tmin, tmax, tavg = int(lens.min()), int(lens.max()), float(lens.mean())
    return {
        "exists": True, "path": path, "rows": len(df),
        "has_cleaned_tokens": has_tokens,
        "nonempty_token_rows": nonempty,
        "token_len_min": tmin, "token_len_max": tmax, "token_len_avg": tavg
    }

comments_audit = _audit(COMMENTS_PATH)
captions_audit = _audit(CAPTIONS_PATH)

print("[Audit] cleaned_comments.csv ->", comments_audit)
print("[Audit] cleaned_captions.csv ->", captions_audit)

# ---------------- After loading CSVs, proceed with your workflow ----------------
import os, ast, json
from typing import List, Tuple, Dict
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Utilities ----------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def savefig(path, dpi=300):
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()

def parse_tokens(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            # whitespace fallback to avoid crash
            return [t for t in x.split() if t]
    return []

def load_cleaned_comments(path: str, tokens_col="cleaned_tokens", time_cols=("published_at","timestamp","time")) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if tokens_col not in df.columns:
        raise KeyError(f"Missing required column '{tokens_col}'. Columns found: {list(df.columns)}")
    df[tokens_col] = df[tokens_col].apply(parse_tokens)
    df = df[df[tokens_col].apply(lambda ts: isinstance(ts, list) and len(ts)>0)].copy()
    # Optional chronological sort if a known time column exists
    for c in time_cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.sort_values(c, kind="stable").reset_index(drop=True)
                break
            except Exception:
                pass
    df.reset_index(drop=True, inplace=True)
    return df

def split_into_chunks(df: pd.DataFrame, k: int = 5) -> List[pd.DataFrame]:
    # Equal-sized chunks by row order (proxy for time if no timestamp)
    return list(np.array_split(df, k))

def transactions_from_chunk(chunk: pd.DataFrame, tokens_col="cleaned_tokens", min_len=1) -> List[List[str]]:
    tx = []
    for row in chunk[tokens_col].tolist():
        if isinstance(row, list):
            # deduplicate within a comment's basket for co-occurrence
            basket = list(dict.fromkeys(row))
            if len(basket) >= min_len:
                tx.append(basket)
    return tx

def count_unigrams(transactions: List[List[str]]) -> Counter:
    c = Counter()
    for row in transactions:
        c.update(row)
    return c

def count_cooccurring_pairs(transactions: List[List[str]]) -> Counter:
    c = Counter()
    for row in transactions:
        # unordered, non-adjacent co-occurrence
        c.update(combinations(sorted(set(row)), 2))
    return c

def top_k(counter: Counter, k: int) -> List[Tuple[str,int]]:
    return counter.most_common(k)

# ---------------- Visualization ----------------

def bar_top_counts(name: str, items, title: str, outpng: str):
    if not items: return
    labels = [str(w) for w,_ in items]
    counts = [int(c) for _,c in items]
    y = np.arange(len(labels))[::-1]
    plt.figure(figsize=(9,6))
    plt.barh(y, list(reversed(counts)))
    plt.yticks(y, list(reversed(labels)))
    plt.xlabel("Count")
    plt.title(f"{title} — {name}")
    savefig(outpng)

def plot_lines_over_chunks(series_dict, title: str, outpng: str):
    plt.figure(figsize=(9,6))
    xs = range(1, 1+len(next(iter(series_dict.values()))))
    for label, ys in series_dict.items():
        plt.plot(list(xs), ys, marker="o", label=label)
    plt.xlabel("Chunk")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(list(xs))
    plt.legend()
    savefig(outpng)

# ---------------- Main incremental workflow ----------------

def main():
    ROOT = ensure_dir("Lab3_Incremental_PatternMining")
    FIG = ensure_dir(os.path.join(ROOT, "figures"))
    EXP = ensure_dir(os.path.join(ROOT, "exports"))
    REP = ensure_dir(os.path.join(ROOT, "reports"))

    csv_path = COMMENTS_PATH  # "/mnt/data/cleaned_comments.csv"
    tokens_col = "cleaned_tokens"

    print("[Load] Reading:", csv_path)
    df = load_cleaned_comments(csv_path, tokens_col=tokens_col)
    print(f"[Load] Rows after cleaning (non-empty tokens): {len(df)}")

    # Split into five chunks
    chunks = split_into_chunks(df, k=5)
    print("[Info] Chunk sizes:", [len(c) for c in chunks])

    per_chunk_uni_counters = []
    per_chunk_pair_counters = []

    # Process each chunk
    for i, chunk in enumerate(chunks, start=1):
        name = f"Chunk {i}"
        tx = transactions_from_chunk(chunk, tokens_col=tokens_col, min_len=1)

        uni = count_unigrams(tx)
        pairs = count_cooccurring_pairs(tx)

        per_chunk_uni_counters.append(uni)
        per_chunk_pair_counters.append(pairs)

        # Save top-10 lists
        top10_uni = top_k(uni, 10)
        top10_pairs = top_k(pairs, 10)
        pd.DataFrame(top10_uni, columns=["token","count"]).to_csv(os.path.join(EXP, f"top10_unigrams_chunk{i}.csv"), index=False)
        pd.DataFrame([{"a":a,"b":b,"count":c} for (a,b),c in top10_pairs], columns=["a","b","count"]).to_csv(os.path.join(EXP, f"top10_pairs_chunk{i}.csv"), index=False)

        # Plots
        bar_top_counts(name, [(w,c) for w,c in top10_uni], "Top 10 Unigrams", os.path.join(FIG, f"top10_unigrams_chunk{i}.png"))
        pair_labels = [("{} • {}".format(a,b), c) for (a,b),c in top10_pairs]
        bar_top_counts(name, pair_labels, "Top 10 Co-occurring Pairs", os.path.join(FIG, f"top10_pairs_chunk{i}.png"))

    # -------- Build tracking tables across chunks --------
    TRACK_N = 10  # adjust to 5 if you want just the top-5

    # Global counters (union of chunks)
    global_uni = Counter()
    global_pairs = Counter()
    for c in per_chunk_uni_counters: global_uni.update(c)
    for c in per_chunk_pair_counters: global_pairs.update(c)

    sel_uni = [w for w,_ in global_uni.most_common(TRACK_N)]
    sel_pairs = [ab for ab,_ in global_pairs.most_common(TRACK_N)]

    # Build frequency series per selection across chunks (0 if missing)
    uni_series = { w: [int(c.get(w, 0)) for c in per_chunk_uni_counters] for w in sel_uni }
    pair_series = { f"{a} • {b}": [int(c.get((a,b), 0)) for c in per_chunk_pair_counters] for (a,b) in sel_pairs }

    # Save wide tables
    pd.DataFrame(uni_series).to_csv(os.path.join(EXP, "unigram_freq_over_chunks.csv"), index=False)
    pd.DataFrame(pair_series).to_csv(os.path.join(EXP, "pair_freq_over_chunks.csv"), index=False)

    # Line plots
    plot_lines_over_chunks(uni_series, "Unigram Frequency Over Chunks", os.path.join(FIG, "unigram_lines_over_chunks.png"))
    plot_lines_over_chunks(pair_series, "Pair Frequency Over Chunks", os.path.join(FIG, "pair_lines_over_chunks.png"))

    # -------- Correlation analysis on a small, interesting subset --------
    # Heuristic: take top 5 unigrams and top 5 pairs by global counts for correlation
    corr_uni_candidates = sel_uni[:5]
    corr_pair_candidates = sel_pairs[:5]

    freq_uni_df = pd.DataFrame({ w: uni_series[w] for w in corr_uni_candidates })
    freq_pair_df = pd.DataFrame({ f"{a} • {b}": pair_series[f"{a} • {b}"] for (a,b) in corr_pair_candidates })

    corr_uni = freq_uni_df.corr() if not freq_uni_df.empty else pd.DataFrame()
    corr_pairs = freq_pair_df.corr() if not freq_pair_df.empty else pd.DataFrame()

    freq_uni_df.to_csv(os.path.join(EXP, "corr_unigrams_input.csv"), index=False)
    freq_pair_df.to_csv(os.path.join(EXP, "corr_pairs_input.csv"), index=False)
    corr_uni.to_csv(os.path.join(EXP, "corr_unigrams_matrix.csv"), index=False)
    corr_pairs.to_csv(os.path.join(EXP, "corr_pairs_matrix.csv"), index=False)

    # Simple text report scaffold
    report = f"""# Incremental Mining — Reflection

Data file: {os.path.abspath(csv_path)}
Rows used: {len(df)}

- Chunk sizes: {[len(c) for c in chunks]}
- Correlation (unigrams, top 5):\n{corr_uni.to_string() if not corr_uni.empty else "(not enough data)"}\n
- Correlation (pairs, top 5):\n{corr_pairs.to_string() if not corr_pairs.empty else "(not enough data)"}\n

Guidance:
1) Describe visible rises/falls in the line plots (figures/*_lines_over_chunks.png).
2) Pick 3–5 patterns from the correlation inputs and write 2–3 sentences each.
3) Acknowledge limitation: row chunking approximates time unless truly timestamp-sorted.
"""
    with open(os.path.join(REP, "reflection_incremental.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # Save a summary index for easy navigation
    index = {
        "chunk_sizes": [len(c) for c in chunks],
        "selected_unigrams": corr_uni_candidates,
        "selected_pairs": [list(t) for t in corr_pair_candidates],
        "exports_dir": os.path.abspath(EXP),
        "figures_dir": os.path.abspath(FIG),
        "reports_dir": os.path.abspath(REP)
    }
    with open(os.path.join(ROOT, "OUTPUT_INDEX_INCREMENTAL.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("[Done] Outputs written under:", os.path.abspath(ROOT))

if __name__ == "__main__":
    main()