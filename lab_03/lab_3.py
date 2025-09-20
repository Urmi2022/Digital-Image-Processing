
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg)
# -*- coding: utf-8 -*-
"""
Lab 3 — Pattern Discovery from Cleaned YouTube Text
Frequent Word Co-occurrence & Contextual Insights

This script executes Tasks A–G end-to-end with reproducible outputs and 300 DPI figures.

INPUTS (update paths if needed):
- /mnt/data/cleaned_comments.csv  (must have 'cleaned_tokens' column of lists or list-like strings)
- /mnt/data/cleaned_captions.csv  (must have 'cleaned_tokens' column)

OUTPUT ROOT:
- Lab3_PatternMining/
    comments/, captions/, merged/
    exports/ (CSV tables)
    figures/ (PNG images)
    reports/ (insights, reflection)

Notes:
- Handles 'cleaned_tokens' stored as Python-list strings via ast.literal_eval.
- Auto-chunks a single very-long captions token list into multiple baskets, so Apriori/co-occurrence are meaningful.
- Apriori (mlxtend) is optional. If not available, the script will attempt to pip install; if offline, it will skip Apriori-only plots and CSVs gracefully.
"""

import os, ast, itertools, warnings, json, sys
from collections import Counter
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional imports (handled gracefully)
try:
    import networkx as nx
    _NX_OK = True
except Exception:
    _NX_OK = False

try:
    from wordcloud import WordCloud
    _WC_OK = True
except Exception:
    _WC_OK = False

# ---------------------- Utility ----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def savefig(path, dpi=300, bbox_inches="tight"):
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def safe_literal_eval_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            # Fallback: split on whitespace if literal_eval fails
            return [t for t in x.strip().split() if t]
    return [] if pd.isna(x) else (list(x) if hasattr(x, '__iter__') else [])

_MINIMAL_STOPWORDS = {
    'the','a','an','and','or','of','to','in','on','for','at','by','with','from','as','is',
    'are','was','were','be','been','being','it','its','this','that','these','those','i','you',
    'he','she','they','we','me','him','her','them','my','your','our','their','but','if','so',
    'than','then','too','very','not','no','yes','do','does','did','done','have','has','had',
    'can','could','should','would','will','just','also','about','into','over','more','most',
    'other','some','such','up','out','what','which','who','whom','when','where','why','how'
}

def load_dataset(csv_path: str, tokens_col: str = "cleaned_tokens") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    if tokens_col not in df.columns:
        raise KeyError(f"Expected column '{tokens_col}' not found in {csv_path}. Found: {list(df.columns)}")
    df[tokens_col] = df[tokens_col].apply(safe_literal_eval_list)
    # Drop rows with empty tokens
    df = df[df[tokens_col].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def build_transactions(df: pd.DataFrame, tokens_col: str = "cleaned_tokens",
                       min_len: int = 3, dedup: bool = True) -> List[List[str]]:
    tx = []
    for lst in df[tokens_col].tolist():
        if not isinstance(lst, list):
            continue
        basket = list(dict.fromkeys(lst)) if dedup else lst[:]  # preserve order while removing dups
        if len(basket) >= min_len:
            tx.append(basket)
    return tx

def stats_basket_lengths(transactions: List[List[str]]):
    lengths = [len(t) for t in transactions]
    if not lengths:
        return {"count": 0, "avg": 0, "min": 0, "max": 0}
    return {
        "count": len(lengths),
        "avg": float(np.mean(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }

def plot_hist_lengths(transactions: List[List[str]], outpng: str):
    lengths = [len(t) for t in transactions]
    if not lengths:
        warnings.warn("No transactions to plot.")
        return
    plt.figure(figsize=(8,5))
    plt.hist(lengths, bins=min(50, max(10, int(np.sqrt(len(lengths))))) )
    plt.xlabel("Basket length (# tokens)")
    plt.ylabel("Frequency (# baskets)")
    plt.title("Histogram of Basket Lengths")
    savefig(outpng)

def top_unigrams(transactions: List[List[str]], k: int = 20) -> Tuple[List[Tuple[str, int]], Counter]:
    cnt = Counter(itertools.chain.from_iterable(transactions))
    return cnt.most_common(k), cnt

def plot_top_unigrams(unigram_counter: Counter, k: int, outpng: str):
    items = unigram_counter.most_common(k)
    if not items:
        return
    words, freqs = zip(*items)
    y = np.arange(len(words))[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(y, list(freqs)[::-1])
    plt.yticks(y, list(words)[::-1])
    plt.xlabel("Frequency")
    plt.title(f"Top {k} Unigrams")
    savefig(outpng)

def pair_cooccurrence(transactions: List[List[str]]) -> Counter:
    pair_counter = Counter()
    for basket in transactions:
        # unique pairs from this basket
        unique_items = list(dict.fromkeys(basket))
        for a, b in itertools.combinations(sorted(unique_items), 2):
            pair_counter[(a,b)] += 1
    return pair_counter

def filter_pairs_by_count(pair_counter: Counter, min_count: int = 3) -> Counter:
    return Counter({k:v for k,v in pair_counter.items() if v >= min_count})

def plot_top_pairs(pair_counter: Counter, k: int, outpng: str):
    items = pair_counter.most_common(k)
    if not items:
        return
    pairs = [f"{a} • {b}" for (a,b),_ in items]
    counts = [c for _,c in items]
    y = np.arange(len(pairs))[::-1]
    plt.figure(figsize=(9,7))
    plt.barh(y, list(counts)[::-1])
    # FIX: proper reverse indexing
    plt.yticks(y, list(pairs)[::-1])
    plt.xlabel("Co-occurrence Count (# baskets)")
    plt.title(f"Top {k} Co-occurring Pairs")
    savefig(outpng)

def export_pairs(pair_counter: Counter, csv_out: str):
    ensure_dir(os.path.dirname(csv_out))
    rows = [{"token_a": a, "token_b": b, "count": c} for (a,b), c in pair_counter.items()]
    pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(csv_out, index=False)

def remove_stopwords_from_pairs(pair_counter: Counter, stopwords: set) -> Counter:
    return Counter({(a,b):c for (a,b),c in pair_counter.items() if a not in stopwords and b not in stopwords})

def graph_cooccurrence(pair_counter: Counter, min_count: int, outpng: str, max_nodes: int = 80):
    if not _NX_OK:
        warnings.warn("networkx not available; skipping graph.")
        return
    # Build graph using threshold
    edges = [ (a,b,c) for (a,b), c in pair_counter.items() if c >= min_count ]
    if not edges:
        return
    # Limit nodes to top-degree subset to avoid hairball
    G = nx.Graph()
    for a,b,c in edges:
        G.add_edge(a,b, weight=c)
    if G.number_of_nodes() > max_nodes:
        degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = set([n for n,_ in degrees])
        G = G.subgraph(keep).copy()

    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(max(1,G.number_of_nodes())))
    plt.figure(figsize=(10,8))
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    degrees = dict(G.degree)
    sizes = [300 + 40*degrees[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=[0.5 + 0.2*w for w in weights], alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
    plt.title(f"Co-occurrence Graph (edges ≥ {min_count})")
    savefig(outpng)

def try_wordcloud(unigram_counter: Counter, outpng: str, max_words: int = 200):
    if not _WC_OK:
        warnings.warn("wordcloud not available; skipping word cloud.")
        return
    wc = WordCloud(width=1200, height=800, background_color="white", max_words=max_words)
    wc.generate_from_frequencies(unigram_counter)
    plt.figure(figsize=(10,7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud (Most Frequent Items)")
    savefig(outpng)

def ensure_mlxtend():
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        return True
    except Exception:
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mlxtend"])
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
            return True
        except Exception as e:
            warnings.warn(f"mlxtend not available and could not be installed ({e}). Apriori steps will be skipped.")
            return False

def run_apriori_block(transactions: List[List[str]], supports: List[float], outdir: str,
                      min_rule_conf: float = 0.6, min_rule_lift: float = 1.2,
                      label: str = ""):
    ok = ensure_mlxtend()
    if not ok or not transactions:
        return
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    ensure_dir(outdir)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    for s in supports:
        fis = apriori(onehot, min_support=s, use_colnames=True)
        if fis.empty:
            continue
        fis["length"] = fis["itemsets"].apply(lambda x: len(x))
        fis = fis.sort_values("support", ascending=False)
        fis_path = os.path.join(outdir, f"frequent_itemsets_s{s:.2f}{label}.csv")
        fis.to_csv(fis_path, index=False)

        # Top 2- and 3-itemsets by support
        for L in [2,3]:
            subset = fis[fis["length"]==L].nlargest(10, "support")
            if not subset.empty:
                names = subset["itemsets"].apply(lambda s: " • ".join(sorted(list(s)))).tolist()
                supports_vals = subset["support"].tolist()
                y = np.arange(len(names))[::-1]
                plt.figure(figsize=(9,6))
                plt.barh(y, list(supports_vals)[::-1])
                # FIX: proper reverse indexing for ytick labels
                plt.yticks(y, list(names)[::-1])
                plt.xlabel("Support")
                plt.title(f"Top {min(10,len(names))} {L}-itemsets (support≥{s:.2f})")
                savefig(os.path.join(outdir, f"top_{L}itemsets_s{s:.2f}{label}.png"))

        rules = association_rules(fis, metric="confidence", min_threshold=0.0)
        if rules.empty:
            continue
        rules["antecedents"] = rules["antecedents"].apply(lambda s: sorted(list(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: sorted(list(s)))
        rules_f = rules[(rules["confidence"] >= min_rule_conf) & (rules["lift"] >= min_rule_lift)].copy()
        rules_path = os.path.join(outdir, f"association_rules_s{s:.2f}{label}.csv")
        rules_f.sort_values(["confidence","lift","support"], ascending=False).to_csv(rules_path, index=False)

        plt.figure(figsize=(7,6))
        plt.scatter(rules["support"], rules["confidence"], alpha=0.5)
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title(f"Rules: Support vs Confidence (s≥{s:.2f})")
        savefig(os.path.join(outdir, f"support_vs_confidence_s{s:.2f}{label}.png"))

def chunk_single_long_list(df: pd.DataFrame, tokens_col="cleaned_tokens", chunk_size=30, step=30) -> pd.DataFrame:
    """
    If df has a single row with a very long token list, split it into pseudo-baskets.
    """
    if len(df) == 1 and isinstance(df.iloc[0][tokens_col], list) and len(df.iloc[0][tokens_col]) >= chunk_size*2:
        tokens = df.iloc[0][tokens_col]
        baskets = []
        for i in range(0, len(tokens), step):
            window = tokens[i:i+chunk_size]
            if len(window) >= 3:
                baskets.append(window)
        new_df = pd.DataFrame({tokens_col: baskets})
        print(f"[Info] Captions had one long list; auto-chunked into {len(new_df)} baskets of size≈{chunk_size}.")
        return new_df
    return df

def apply_min_token_length(transactions: List[List[str]], min_chars: int = 4) -> List[List[str]]:
    out = []
    for b in transactions:
        nb = [t for t in b if len(t) >= min_chars]
        if len(nb) >= 3:
            out.append(nb)
    return out

def apply_stemming(transactions: List[List[str]]):
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return [ [stemmer.stem(t) for t in b] for b in transactions ]
    except Exception:
        warnings.warn("NLTK not available; skipping stemming variation.")
        return transactions

def pipeline_run(name: str, df: pd.DataFrame, root_out: str, tokens_col: str = "cleaned_tokens",
                 min_pair_count: int = 3):
    """
    Runs Tasks A–D & F on a given dataframe and writes results under root_out/name.
    """
    outdir = ensure_dir(os.path.join(root_out, name))
    figs = ensure_dir(os.path.join(outdir, "figures"))
    exports = ensure_dir(os.path.join(outdir, "exports"))

    # Baskets
    transactions = build_transactions(df, tokens_col=tokens_col, min_len=3, dedup=True)

    # Stats
    stats = stats_basket_lengths(transactions)
    with open(os.path.join(outdir, "basket_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[{name}] baskets={stats['count']}, avg_len={stats['avg']:.2f}, min={stats['min']}, max={stats['max']}")

    # Plots: histogram
    plot_hist_lengths(transactions, os.path.join(figs, "basket_hist.png"))

    # Unigrams
    top20, uni_cnt = top_unigrams(transactions, k=20)
    pd.DataFrame(top20, columns=["token","frequency"]).to_csv(os.path.join(exports, "top20_unigrams.csv"), index=False)
    plot_top_unigrams(uni_cnt, k=20, outpng=os.path.join(figs, "top20_unigrams.png"))
    try_wordcloud(uni_cnt, os.path.join(figs, "wordcloud.png"))

    # Pairwise co-occurrence
    pairs = pair_cooccurrence(transactions)
    export_pairs(pairs, os.path.join(exports, "pair_cooccurrence_raw.csv"))
    pairs_f = filter_pairs_by_count(pairs, min_count=min_pair_count)
    plot_top_pairs(pairs_f, k=20, outpng=os.path.join(figs, f"top_pairs_count_ge_{min_pair_count}.png"))
    # Stopword filtering
    pairs_wo_stop = remove_stopwords_from_pairs(pairs_f, _MINIMAL_STOPWORDS)
    export_pairs(pairs_wo_stop, os.path.join(exports, "pair_cooccurrence_no_stopwords.csv"))
    # Graph
    graph_cooccurrence(pairs_wo_stop, min_count=min_pair_count, outpng=os.path.join(figs, "cooccurrence_graph.png"))

    # Apriori (multiple supports)
    supports_primary = [0.30, 0.20, 0.10]
    run_apriori_block(transactions, supports_primary, exports, label=f"_{name}")

    # Return key artifacts for variations
    return transactions, uni_cnt, pairs_wo_stop

def pipeline_variations(name: str, base_transactions: List[List[str]], root_out: str):
    """
    Runs E: support variations, stemming, min token length variations.
    """
    outdir = ensure_dir(os.path.join(root_out, name, "variations"))
    # Supports 0.15 and 0.05
    run_apriori_block(base_transactions, [0.15, 0.05], outdir, label="_var_support")

    # Stemming variation
    stem_tx = apply_stemming(base_transactions)
    run_apriori_block(stem_tx, [0.20], outdir, label="_stemming")

    # Remove tokens under 4 chars
    long_tx = apply_min_token_length(base_transactions, min_chars=4)
    run_apriori_block(long_tx, [0.20], outdir, label="_minlen4")

def write_reflection_template(root_out: str):
    reports = ensure_dir(os.path.join(root_out, "reports"))
    md = """# Lab 3 — Reflection (Template)

**Q1. What’s a surprising pair you found?**  
<Write your answer here and include the co-occurrence count and/or the rule’s support/confidence if relevant.>

**Q2. What was hard about finding patterns?**  
<Discuss data issues (noise, imbalance), parameter sensitivity (support/confidence), or computational constraints.>

**Q3. What would you explore next?**  
<Propose follow-ups: topic-specific stopwords, POS filtering, lemmatization vs. stemming, merging vs. separate datasets, windowed co-occurrence by time, etc.>

---

### Sample Comparison (for your self-check)
- You might find patterns like "online • class", "exam • stress", or "ai • tools".
- Strong rules often involve domain terms co-appearing with qualifiers (e.g., "online, class" ⇒ "lecture").
- Be wary of trivial pairs dominated by stopwords or boilerplate.

> Remember to label each insight with metrics (support, confidence, lift) when applicable.
"""
    with open(os.path.join(reports, "reflection_template.md"), "w", encoding="utf-8") as f:
        f.write(md)

def main():
    ROOT = ensure_dir("Lab3_PatternMining")
    # ----- Load data (A.1-3, sanity checks included) -----
    comments_csv = "cleaned_comments.csv"
    captions_csv = "cleaned_captions.csv"

    print("[Load] Reading comments...")
    df_comments = load_dataset(comments_csv)
    print(f"[Load] Comments rows after cleaning: {len(df_comments)}")

    print("[Load] Reading captions...")
    df_captions = load_dataset(captions_csv)
    # If captions is a single long list, chunk it to make multiple baskets
    df_captions = chunk_single_long_list(df_captions, chunk_size=30, step=30)
    print(f"[Load] Captions rows after cleaning/chunking: {len(df_captions)}")

    # ----- Comments pipeline -----
    tx_comments, uni_c_comments, pairs_comments = pipeline_run("comments", df_comments, ROOT)

    # ----- Captions pipeline -----
    tx_captions, uni_c_captions, pairs_captions = pipeline_run("captions", df_captions, ROOT)

    # ----- Merged pipeline (E.36-38, F.43) -----
    merged_df = pd.concat([df_comments[["cleaned_tokens"]], df_captions[["cleaned_tokens"]]], ignore_index=True)
    tx_merged, uni_c_merged, pairs_merged = pipeline_run("merged", merged_df, ROOT)

    # ----- Variations (E.31-35, 37-38) -----
    pipeline_variations("comments", tx_comments, ROOT)
    pipeline_variations("captions", tx_captions, ROOT)
    pipeline_variations("merged", tx_merged, ROOT)

    # ----- Wrap-up (G.46-53) -----
    exports = ensure_dir(os.path.join(ROOT, "exports"))
    summary = {
        "paths": {
            "comments": os.path.abspath(os.path.join(ROOT, "comments")),
            "captions": os.path.abspath(os.path.join(ROOT, "captions")),
            "merged": os.path.abspath(os.path.join(ROOT, "merged")),
        },
        "notes": [
            "All frequent itemsets and association rules CSVs (if mlxtend available) are under each dataset's 'exports' folder.",
            "All figures are saved under each dataset's 'figures' folder.",
            "Reflection template placed under Lab3_PatternMining/reports/."
        ]
    }
    with open(os.path.join(exports, "OUTPUT_INDEX.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_reflection_template(ROOT)

    print("\n[Done] Lab3 outputs written under:", os.path.abspath(ROOT))

if __name__ == "__main__":
    main()