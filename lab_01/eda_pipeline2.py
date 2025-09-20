# -*- coding: utf-8 -*-
"""
Lab EDA Pipeline
Solves:
- PART 3: Data Processing & Lab Experiments (A.1, A.2, B.1, B.2, B.3)
- PART 4: Checklist & Additional Tasks (basic stats/plots + key questions)

Outputs go to ./EDA/
"""

import os
import re
import glob
import math
import json
from collections import Counter
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Global config
# -----------------------------
COMMENTS_CSV = "comments_all.csv"       # your 22k comments CSV
CAPTIONS_GLOB = "captions.en.vtt"                 # auto-detect captions file (e.g., *.en.vtt)
EDA_DIR = "EDA"
DPI = 300

os.makedirs(EDA_DIR, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def save_txt(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def safe_figsave(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()

# -----------------------------
# PART 3.A.1 — Extracting Comments from comments.txt (manual copy)
# (We include it exactly as spec + a variant that returns lines if file exists)
# -----------------------------
def load_raw_comments_from_txt(filepath="comments.txt"):
    """
    Matches your provided skeleton:
    - ignore empty lines
    - ignore lines ending with 'ago'
    - ignore lines equal to 'reply'
    """
    comments = []
    if not os.path.exists(filepath):
        return comments  # if not present, we skip this path
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.endswith("ago") and line.lower() != "reply":
                comments.append(line)
    return comments

# -----------------------------
# PART 3.A (CSV route): Load comments from your CSV (preferred)
# -----------------------------
def load_comments_csv(csv_path=COMMENTS_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # ensure text and numeric types
    df["text"] = df["text"].astype(str)
    if "like_count" in df.columns:
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    else:
        df["like_count"] = 0
    # published may be unix seconds; convert if plausible
    if "published" in df.columns:
        try:
            df["published_dt"] = pd.to_datetime(df["published"], unit="s", errors="coerce")
        except Exception:
            df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    else:
        df["published_dt"] = pd.NaT
    # mark replies/top-level based on parent_id if present
    if "parent_id" in df.columns:
        df["is_reply"] = df["parent_id"].apply(lambda x: "Reply" if str(x).strip().lower() != "root" else "Top-level")
    else:
        df["is_reply"] = "Top-level"
    # basic lengths
    df["text_length"] = df["text"].str.len()
    return df

# -----------------------------
# PART 3.A.2 — Extract caption text from .vtt
# - robust: use webvtt if available for timestamps; fallback to simple line filter
# -----------------------------
def load_vtt_captions_auto(vtt_glob=CAPTIONS_GLOB):
    """
    Returns DataFrame with columns:
    ['start', 'end', 'text', 'start_sec', 'end_sec', 'mid_sec', 'text_length']
    If webvtt not available or timestamps missing, we still return text-only rows.
    """
    vtt_files = sorted(glob.glob(vtt_glob))
    if not vtt_files:
        return pd.DataFrame(columns=["start","end","text","start_sec","end_sec","mid_sec","text_length"])

    vtt_path = vtt_files[0]  # take the first match
    rows = []

    try:
        import webvtt
        for cue in webvtt.read(vtt_path):
            txt = (cue.text or "").replace("\r", " ").replace("\n", " ").strip()
            # parse hh:mm:ss.mmm -> seconds
            def hms_to_sec(hhmmss):
                # webvtt returns 'HH:MM:SS.mmm'
                h, m, s = hhmmss.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            try:
                s_sec = hms_to_sec(cue.start)
                e_sec = hms_to_sec(cue.end)
            except Exception:
                s_sec, e_sec = math.nan, math.nan
            mid = (s_sec + e_sec) / 2 if pd.notna(s_sec) and pd.notna(e_sec) else math.nan
            rows.append({
                "start": cue.start,
                "end": cue.end,
                "text": txt,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "mid_sec": mid,
                "text_length": len(txt)
            })
    except Exception:
        # Fallback: simple line filter like your skeleton (no timestamps)
        with open(vtt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "-->" not in line and line and not line.isdigit() and "WEBVTT" not in line:
                    txt = line.replace("\r", " ").strip()
                    rows.append({
                        "start": None, "end": None,
                        "text": txt,
                        "start_sec": math.nan, "end_sec": math.nan, "mid_sec": math.nan,
                        "text_length": len(txt)
                    })

    df_caps = pd.DataFrame(rows)
    return df_caps

# -----------------------------
# PART 3.B.1 — Histogram of Lengths (Captions vs Comments)
# -----------------------------
def plot_hist_lengths(df_caps, df_comments):
    cap_len = df_caps["text_length"].dropna().tolist()
    com_len = df_comments["text_length"].dropna().tolist()

    plt.figure(figsize=(9,6))
    # Captions
    plt.hist(cap_len, bins=40, alpha=0.7, label="Captions")
    # Comments
    plt.hist(com_len, bins=40, alpha=0.7, label="Comments")
    plt.legend()
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Caption vs. Comment Lengths")
    safe_figsave(os.path.join(EDA_DIR, "hist_lengths_captions_vs_comments.png"))

# -----------------------------
# PART 3.B.2 — Vocabulary Diversity (Type-Token Ratio)
# -----------------------------
STOPWORDS = set([
    'the','and','a','is','in','to','of','that','it','on','for','with','as','this','was','but','are','not','be','at',
    'by','an','if','or','from','so','we'
])
def tokenize_simple(lines):
    words = []
    for line in lines:
        # strip punctuation lightly
        w = re.findall(r"[A-Za-z0-9_']+", str(line).lower())
        words.extend(w)
    return words

def type_token_ratio(lines):
    words = tokenize_simple(lines)
    unique = set(words)
    return (len(unique) / len(words)) if words else 0.0

def compute_ttr_and_save(df_caps, df_comments):
    cap_ttr = type_token_ratio(df_caps["text"].tolist())
    com_ttr = type_token_ratio(df_comments["text"].tolist())
    out = f"Caption TTR: {cap_ttr:.4f}\nComment TTR: {com_ttr:.4f}\n"
    save_txt(os.path.join(EDA_DIR, "ttr.txt"), out)

    # also bar chart for TTR
    plt.figure(figsize=(5,4))
    plt.bar(["Captions","Comments"], [cap_ttr, com_ttr])
    plt.ylabel("Type-Token Ratio")
    plt.title("Vocabulary Diversity (TTR)")
    safe_figsave(os.path.join(EDA_DIR, "ttr_bar.png"))

# -----------------------------
# PART 3.B.3 — Top-N Word Frequency (after stopword removal)
# -----------------------------
def top_n_words(lines, n=20):
    words = [w for w in tokenize_simple(lines) if w not in STOPWORDS]
    return Counter(words).most_common(n)

def top_words_plots(df_caps, df_comments, n=20):
    cap_top = top_n_words(df_caps["text"].tolist(), n=n)
    com_top = top_n_words(df_comments["text"].tolist(), n=n)

    # Save as CSV
    pd.DataFrame(cap_top, columns=["word","count"]).to_csv(os.path.join(EDA_DIR, "top_words_captions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(com_top, columns=["word","count"]).to_csv(os.path.join(EDA_DIR, "top_words_comments.csv"), index=False, encoding="utf-8-sig")

    # Bar plots (captions)
    plt.figure(figsize=(10,6))
    words = [w for w,_ in cap_top]
    counts = [c for _,c in cap_top]
    plt.barh(words[::-1], counts[::-1])
    plt.title("Top Words — Captions (stopwords removed)")
    plt.xlabel("Count")
    safe_figsave(os.path.join(EDA_DIR, "top_words_captions.png"))

    # Bar plots (comments)
    plt.figure(figsize=(10,6))
    words = [w for w,_ in com_top]
    counts = [c for _,c in com_top]
    plt.barh(words[::-1], counts[::-1])
    plt.title("Top Words — Comments (stopwords removed)")
    plt.xlabel("Count")
    safe_figsave(os.path.join(EDA_DIR, "top_words_comments.png"))

# -----------------------------
# PART 4 — Checklist & Additional Tasks
# 1) Basic Stats & Visualizations
#    - captions vs # comments
#    - caption line lengths stats & histogram
#    - scatter: caption length vs timestamp
#    - scatter: comment length vs likes
# 2) Key Questions
#    - longer captions near beginning/end?
#    - many short messages?
#    - peaks in commenting activity vs caption timestamps? (parse timecodes in comments)
# -----------------------------
def basic_stats(df_caps, df_comments):
    stats_text = []
    stats_text.append(f"Total caption lines: {len(df_caps)}")
    stats_text.append(f"Total comments: {len(df_comments)}")
    stats_text.append("")

    # Caption lengths
    if len(df_caps):
        desc = df_caps["text_length"].describe(percentiles=[.1,.25,.5,.75,.9,.95]).to_string()
        stats_text.append("Caption length stats:\n" + desc + "\n")

    # Comment lengths
    desc_c = df_comments["text_length"].describe(percentiles=[.1,.25,.5,.75,.9,.95]).to_string()
    stats_text.append("Comment length stats:\n" + desc_c + "\n")

    # Short messages share (<= 20 chars)
    short_pct = (df_comments["text_length"] <= 20).mean() * 100.0
    stats_text.append(f"Short comments (<=20 chars): {short_pct:.2f}%")

    save_txt(os.path.join(EDA_DIR, "basic_stats.txt"), "\n".join(stats_text))

def plot_captions_vs_comments_counts(df_caps, df_comments):
    # tiny bar chart comparing counts
    plt.figure(figsize=(5,4))
    plt.bar(["Captions","Comments"], [len(df_caps), len(df_comments)])
    plt.title("Counts: Captions vs Comments")
    safe_figsave(os.path.join(EDA_DIR, "counts_captions_vs_comments.png"))

def plot_caption_length_hist(df_caps):
    if not len(df_caps): return
    plt.figure(figsize=(9,6))
    plt.hist(df_caps["text_length"].dropna(), bins=40)
    plt.xlabel("Caption Line Length (chars)")
    plt.ylabel("Frequency")
    plt.title("Caption Line Length Distribution")
    safe_figsave(os.path.join(EDA_DIR, "caption_length_hist.png"))

def plot_caption_len_vs_time(df_caps):
    # scatter: caption length vs timestamp (mid_sec)
    if "mid_sec" not in df_caps.columns or df_caps["mid_sec"].isna().all():
        return
    plt.figure(figsize=(12,6))
    plt.scatter(df_caps["mid_sec"], df_caps["text_length"], s=8, alpha=0.4)
    plt.xlabel("Video Time (sec)")
    plt.ylabel("Caption Length (chars)")
    plt.title("Caption Length vs Video Time")
    safe_figsave(os.path.join(EDA_DIR, "caption_length_vs_time_scatter.png"))

    # rolling average to see trend
    c = df_caps[["mid_sec","text_length"]].dropna().sort_values("mid_sec").reset_index(drop=True)
    if len(c) >= 20:
        c["roll_len"] = c["text_length"].rolling(window=30, min_periods=5).mean()
        plt.figure(figsize=(12,6))
        plt.plot(c["mid_sec"], c["roll_len"])
        plt.xlabel("Video Time (sec)")
        plt.ylabel("Caption Length (rolling mean)")
        plt.title("Trend: Caption Length over Time")
        safe_figsave(os.path.join(EDA_DIR, "caption_length_trend.png"))

def plot_comment_len_vs_likes(df_comments):
    plt.figure(figsize=(9,6))
    plt.scatter(df_comments["text_length"], df_comments["like_count"], s=8, alpha=0.4)
    plt.xlabel("Comment Length (chars)")
    plt.ylabel("Likes")
    plt.title("Comment Length vs Likes")
    safe_figsave(os.path.join(EDA_DIR, "comment_length_vs_likes_scatter.png"))

def hhmmss_to_seconds(hh, mm, ss):
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def mmss_to_seconds(mm, ss):
    return int(mm) * 60 + int(ss)

def extract_timecode_seconds_from_text(text):
    """
    Extract all H:MM:SS or MM:SS occurrences in a comment and return list of seconds.
    """
    t = str(text)
    secs = []

    # H:MM:SS (e.g., 1:23:45 or 00:12:34)
    for m in re.finditer(r"(?<!\d)(\d{1,2}):([0-5]\d):([0-5]\d)(?!\d)", t):
        secs.append(hhmmss_to_seconds(m.group(1), m.group(2), m.group(3)))

    # MM:SS (avoid double-counting those that already matched H:MM:SS)
    for m in re.finditer(r"(?<!\d)([0-5]?\d):([0-5]\d)(?!\d)", t):
        mm = int(m.group(1))
        ss = int(m.group(2))
        # skip cases that look like H:MM:SS (already captured)
        # and skip if mm >= 60 (invalid)
        if 0 <= mm < 60 and 0 <= ss < 60:
            secs.append(mmss_to_seconds(mm, ss))

    return secs

def plot_timecode_mentions_by_minute(df_comments):
    # Build a long list of all timecode seconds mentioned in comments
    all_secs = []
    for txt in df_comments["text"].tolist():
        all_secs.extend(extract_timecode_seconds_from_text(txt))

    if not all_secs:
        save_txt(os.path.join(EDA_DIR, "timecode_mentions_note.txt"),
                 "No timecodes (MM:SS / H:MM:SS) detected in comments.")
        return

    # Count by minute
    minutes = [sec // 60 for sec in all_secs]
    freq = pd.Series(minutes).value_counts().sort_index()

    # Save CSV of counts
    out = pd.DataFrame({"minute": freq.index, "mentions": freq.values})
    out.to_csv(os.path.join(EDA_DIR, "timecode_mentions_by_minute.csv"), index=False, encoding="utf-8-sig")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(out["minute"], out["mentions"], marker="o")
    plt.xlabel("Video Minute (0 = start)")
    plt.ylabel("Mentions in Comments")
    plt.title("Comment Timecode Mentions by Minute")
    safe_figsave(os.path.join(EDA_DIR, "timecode_mentions_by_minute.png"))

def key_questions_analysis(df_caps, df_comments):
    """
    Writes answers to key questions into EDA/key_findings.txt
    - Longer captions near beginning or end?
    - Many short messages?
    - Peaks in comment timecode mentions?
    """
    findings = []

    # Longer captions near beginning/end?
    if "mid_sec" in df_caps.columns and not df_caps["mid_sec"].isna().all():
        total_duration = df_caps["end_sec"].max()
        if pd.notna(total_duration) and total_duration > 0:
            dfc = df_caps.dropna(subset=["mid_sec"]).copy()
            dfc["pos"] = dfc["mid_sec"] / total_duration  # 0..1
            # avg length in first 10% vs last 10%
            first = dfc[dfc["pos"] <= 0.10]["text_length"].mean()
            last  = dfc[dfc["pos"] >= 0.90]["text_length"].mean()
            findings.append(f"Average caption length (first 10%): {first:.1f} chars")
            findings.append(f"Average caption length (last 10%):  {last:.1f} chars")
            findings.append("Interpretation: higher average suggests longer captions in that region.\n")
        else:
            findings.append("Could not determine video duration from VTT; skipping begin/end comparison.\n")
    else:
        findings.append("No caption timestamps available; skipping begin/end length comparison.\n")

    # Many short messages?
    short_share = (df_comments["text_length"] <= 20).mean() * 100.0
    findings.append(f"Short comments (<=20 chars) share: {short_share:.2f}%")
    findings.append("Interpretation: Higher % implies many brief/emoji-style comments.\n")

    # Peaks in commenting activity vs caption timestamps? (via timecodes)
    # We report top-10 minutes with most mentions (if we computed them)
    tc_csv = os.path.join(EDA_DIR, "timecode_mentions_by_minute.csv")
    if os.path.exists(tc_csv):
        tdf = pd.read_csv(tc_csv)
        tdf = tdf.sort_values("mentions", ascending=False).head(10)
        findings.append("Top minutes with most timecode mentions (minute: mentions):")
        findings.extend([f"- {int(r.minute)}: {int(r.mentions)}" for _, r in tdf.iterrows()])
        findings.append("\nInterpretation: spikes likely correspond to notable moments in the video.")
    else:
        findings.append("No timecode mentions detected in comments; cannot compute peaks.")

    save_txt(os.path.join(EDA_DIR, "key_findings.txt"), "\n".join(findings))

# -----------------------------
# MAIN: run all steps in order
# -----------------------------
def main():
    # PART 3.A.1 — Manual comments.txt loader (optional)
    raw_comments_txt = load_raw_comments_from_txt("comments.txt")

    # PART 3.A — Preferred CSV loader for your 22k dataset
    df_comments = load_comments_csv(COMMENTS_CSV)

    # PART 3.A.2 — Captions from VTT
    df_caps = load_vtt_captions_auto(CAPTIONS_GLOB)
    # For the histogram in B.1, we need a text_length column even if timestamps missing
    if "text_length" not in df_caps.columns and "text" in df_caps.columns:
        df_caps["text_length"] = df_caps["text"].astype(str).str.len()

    # Quick sanity summary to EDA/readme.txt
    readme = []
    readme.append(f"Found CSV comments: {len(df_comments)} rows")
    readme.append(f"Found VTT captions: {len(df_caps)} lines")
    readme.append(f"Also found manual comments.txt lines: {len(raw_comments_txt)} (if file present)")
    save_txt(os.path.join(EDA_DIR, "readme.txt"), "\n".join(readme))

    # PART 3.B.1 — Histogram of lengths (captions vs comments)
    if len(df_caps) > 0:
        plot_hist_lengths(df_caps, df_comments)

    # PART 3.B.2 — TTR (vocabulary diversity)
    if len(df_caps) > 0:
        compute_ttr_and_save(df_caps, df_comments)
    else:
        # Still produce TTR for comments-only, and note missing captions
        com_ttr = type_token_ratio(df_comments["text"].tolist())
        save_txt(os.path.join(EDA_DIR, "ttr.txt"), f"(No captions found)\nComment TTR: {com_ttr:.4f}\n")

    # PART 3.B.3 — Top-N words (stopwords removed)
    if len(df_caps) > 0:
        top_words_plots(df_caps, df_comments)
    else:
        # Comments-only version
        com_top = top_n_words(df_comments["text"].tolist(), n=20)
        pd.DataFrame(com_top, columns=["word","count"]).to_csv(os.path.join(EDA_DIR, "top_words_comments.csv"), index=False, encoding="utf-8-sig")
        plt.figure(figsize=(10,6))
        words = [w for w,_ in com_top]
        counts = [c for _,c in com_top]
        plt.barh(words[::-1], counts[::-1])
        plt.title("Top Words — Comments (stopwords removed)")
        plt.xlabel("Count")
        safe_figsave(os.path.join(EDA_DIR, "top_words_comments.png"))

    # PART 4 — Checklist & Additional Tasks: basic stats & plots
    basic_stats(df_caps, df_comments)
    plot_captions_vs_comments_counts(df_caps, df_comments)
    if len(df_caps) > 0:
        plot_caption_length_hist(df_caps)
        plot_caption_len_vs_time(df_caps)
    plot_comment_len_vs_likes(df_comments)

    # PART 4 — Key Questions (including timecode-based activity)
    plot_timecode_mentions_by_minute(df_comments)
    key_questions_analysis(df_caps, df_comments)

    print(f"✅ Done. See the '{EDA_DIR}' folder for outputs.")

if __name__ == "__main__":
    main()