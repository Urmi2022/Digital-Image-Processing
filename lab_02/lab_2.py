import os, re, glob, math
from collections import Counter
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# FOLDERS & CONFIG
# ----------------------------
BASE_DIR = os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "outputs")
EDA_DIR = os.path.join(OUT_DIR, "EDA")
CLEAN_DIR = os.path.join(OUT_DIR, "CLEAN")
TABLES_DIR = os.path.join(OUT_DIR, "TABLES")

for d in (OUT_DIR, EDA_DIR, CLEAN_DIR, TABLES_DIR):
    os.makedirs(d, exist_ok=True)

DPI = 300
COMMENTS_TXT = "comments.txt"          # optional manual text file
COMMENTS_CSV = "comments_all.csv"      # your main 22k CSV
CAPTIONS_VTT = "captions.en.vtt"       # your captions file


import nltk

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)


# ----------------------------
# HELPERS
# ----------------------------
def save_txt(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def figsave(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def is_file(path):
    return os.path.exists(path) and os.path.isfile(path)

# ----------------------------
# 3) PROFILING (quick summaries)
# ----------------------------
def profile_file_quick(path, max_lines=50):
    if not is_file(path): 
        return f"[!] Not found: {path}\n"
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            lines.append(line.rstrip("\n"))
    summary = f"=== Preview of {path} (first {len(lines)} lines) ===\n" + "\n".join(lines) + "\n"
    return summary

# ----------------------------
# 5A) PARSE COMMENTS FROM comments.txt (username / timestamp / comment block)
#     Adapt if your format differs
# ----------------------------
def structure_comments_from_txt(filepath=COMMENTS_TXT):
    if not is_file(filepath):
        return pd.DataFrame(columns=["username","timestamp_text","comment_text"])
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    time_regex = re.compile(r".*(second|minute|hour|day|week|month|year)s?\s+ago.*", re.IGNORECASE)
    comments_data, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        # username on current line + time-ago next line
        if i+1 < len(lines) and time_regex.match(lines[i+1].strip()):
            username = line
            timestamp = lines[i+1].strip()
            comment_text = []
            i += 2
            while i < len(lines) and not (i+1 < len(lines) and time_regex.match(lines[i+1].strip())):
                comment_line = lines[i].strip()
                if comment_line and comment_line.lower() not in ("reply", "...more"):
                    comment_text.append(comment_line)
                i += 1
            if comment_text:
                comments_data.append({
                    "username": username,
                    "timestamp_text": timestamp,
                    "comment_text": " ".join(comment_text)
                })
        else:
            i += 1

    return pd.DataFrame(comments_data)

# ----------------------------
# 5A alt) LOAD COMMENTS FROM CSV (preferred for your 22k)
# ----------------------------
def load_comments_csv(csv_path=COMMENTS_CSV):
    if not is_file(csv_path):
        return pd.DataFrame(columns=["comment_id","parent_id","author","text","published","like_count"])
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    for col in ["comment_id","parent_id","author","text","published","like_count"]:
        if col not in df.columns:
            df[col] = None
    # Types
    df["text"] = df["text"].astype(str)
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    # Timestamp -> datetime (UNIX seconds expected)
    try:
        df["published_dt"] = pd.to_datetime(df["published"], unit="s", errors="coerce")
    except Exception:
        df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    # Lengths
    df["text_length"] = df["text"].str.len()
    # Top-level vs replies
    df["type"] = df["parent_id"].apply(lambda x: "Reply" if str(x).strip().lower() != "root" else "Top-level")
    return df

# ----------------------------
# 5B) PARSE CAPTIONS FROM VTT (webvtt first, fallback to manual)
# ----------------------------
def structure_captions_from_vtt(filepath=CAPTIONS_VTT):
    rows = []
    if not is_file(filepath):
        return pd.DataFrame(columns=["start","end","text","start_sec","end_sec","mid_sec","text_length"])
    try:
        import webvtt
        def hms_to_sec(hhmmss):
            h, m, s = hhmmss.split(":")
            return int(h)*3600 + int(m)*60 + float(s)
        for cue in webvtt.read(filepath):
            txt = (cue.text or "").replace("\r"," ").replace("\n"," ").strip()
            try:
                s_sec = hms_to_sec(cue.start)
                e_sec = hms_to_sec(cue.end)
            except Exception:
                s_sec, e_sec = math.nan, math.nan
            rows.append({
                "start": cue.start,
                "end": cue.end,
                "text": txt,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "mid_sec": (s_sec+e_sec)/2 if (isinstance(s_sec, (int,float)) and isinstance(e_sec, (int,float))) else math.nan,
                "text_length": len(txt)
            })
    except Exception as e:
        # Fallback simple line filter
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "-->" not in line and line and not line.isdigit() and "WEBVTT" not in line:
                    txt = line.replace("\r"," ").strip()
                    rows.append({
                        "start": None, "end": None, "text": txt,
                        "start_sec": math.nan, "end_sec": math.nan, "mid_sec": math.nan,
                        "text_length": len(txt)
                    })
    return pd.DataFrame(rows)

# ----------------------------
# 5C) CLEANING PIPELINE (NLTK)
# ----------------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

STOP_EN = set(stopwords.words("english"))
LEM = WordNetLemmatizer()
STEM = PorterStemmer()

def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"\[.*?\]", "", s)         # remove [tags]
    s = re.sub(r"[^a-z\s]", " ", s)       # keep letters + space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    return word_tokenize(s)

def remove_stop(tokens):
    return [w for w in tokens if w not in STOP_EN and len(w) > 2]

def lemmatize(tokens):
    return [LEM.lemmatize(w) for w in tokens]

def stem(tokens):
    return [STEM.stem(w) for w in tokens]

def clean_pipeline(text: str, use_lemmatization=True):
    s = normalize_text(text or "")
    toks = tokenize(s)
    toks = remove_stop(toks)
    toks = lemmatize(toks) if use_lemmatization else stem(toks)
    return toks

# ----------------------------
# 5B.3) CAPTIONS SENTENCE LIST (if you want sentence-level)
# ----------------------------
def captions_to_sentences(df_caps):
    # joins all text then splits by punctuation into sentences
    full_text = " ".join(df_caps["text"].astype(str).tolist())
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return pd.DataFrame({"caption_sentence": sentences})

# ----------------------------
# EDA PLOTS (matplotlib only, one chart per figure)
# ----------------------------
def plot_hist_lengths_captions_vs_comments(df_caps, df_comm):
    plt.figure(figsize=(8,5))
    plt.hist(df_caps["text_length"].dropna(), bins=40, alpha=0.7, label="Captions")
    plt.hist(df_comm["text_length"].dropna(), bins=40, alpha=0.5, label="Comments")
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Caption vs Comment Lengths")
    plt.legend()
    figsave(os.path.join(EDA_DIR, "hist_lengths_captions_vs_comments.png"))

def type_token_ratio(lines):
    words = []
    for line in lines:
        words += re.findall(r"[a-z0-9']+", (line or "").lower())
    return (len(set(words)) / len(words)) if words else 0.0

def plot_ttr_bars(df_caps, df_comm):
    cap_ttr = type_token_ratio(df_caps["text"].astype(str).tolist())
    com_ttr = type_token_ratio(df_comm["text"].astype(str).tolist())
    with open(os.path.join(EDA_DIR, "ttr.txt"), "w", encoding="utf-8") as f:
        f.write(f"Caption TTR: {cap_ttr:.4f}\nComment TTR: {com_ttr:.4f}\n")
    plt.figure(figsize=(5,4))
    plt.bar(["Captions","Comments"], [cap_ttr, com_ttr])
    plt.ylabel("Type-Token Ratio")
    plt.title("Vocabulary Diversity")
    figsave(os.path.join(EDA_DIR, "ttr_bar.png"))

def top_n_words(lines, n=20):
    words = []
    for line in lines:
        for w in re.findall(r"[a-z']+", (line or "").lower()):
            if w not in STOP_EN and len(w) > 2:
                words.append(w)
    return Counter(words).most_common(n)

def plot_top_words(df_caps, df_comm, n=20):
    cap_top = top_n_words(df_caps["text"].astype(str).tolist(), n)
    com_top = top_n_words(df_comm["text"].astype(str).tolist(), n)
    pd.DataFrame(cap_top, columns=["word","count"]).to_csv(os.path.join(TABLES_DIR,"top_words_captions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(com_top, columns=["word","count"]).to_csv(os.path.join(TABLES_DIR,"top_words_comments.csv"), index=False, encoding="utf-8-sig")

    # captions barh
    plt.figure(figsize=(8,6))
    words = [w for w,_ in cap_top][::-1]
    counts = [c for _,c in cap_top][::-1]
    plt.barh(words, counts)
    plt.xlabel("Count")
    plt.title("Top Words — Captions (stopwords removed)")
    figsave(os.path.join(EDA_DIR, "top_words_captions.png"))

    # comments barh
    plt.figure(figsize=(8,6))
    words = [w for w,_ in com_top][::-1]
    counts = [c for _,c in com_top][::-1]
    plt.barh(words, counts)
    plt.xlabel("Count")
    plt.title("Top Words — Comments (stopwords removed)")
    figsave(os.path.join(EDA_DIR, "top_words_comments.png"))

def plot_comment_length_vs_likes(df_comm):
    if "like_count" not in df_comm.columns: return
    plt.figure(figsize=(7,5))
    plt.scatter(df_comm["text_length"], df_comm["like_count"], s=8, alpha=0.4)
    plt.xlabel("Comment Length (chars)")
    plt.ylabel("Likes")
    plt.title("Comment Length vs Likes")
    figsave(os.path.join(EDA_DIR, "comment_length_vs_likes_scatter.png"))

def plot_comments_over_time(df_comm):
    if "published_dt" not in df_comm.columns: return
    tmp = df_comm.dropna(subset=["published_dt"]).copy()
    if tmp.empty: return
    daily = tmp.groupby(tmp["published_dt"].dt.date).size()
    plt.figure(figsize=(10,5))
    daily.plot()
    plt.xlabel("Date")
    plt.ylabel("Comments per Day")
    plt.title("Comments Over Time")
    figsave(os.path.join(EDA_DIR, "comments_over_time.png"))

def plot_caption_length_vs_time(df_caps):
    # needs mid_sec (from VTT)
    if "mid_sec" not in df_caps.columns or df_caps["mid_sec"].isna().all(): return
    plt.figure(figsize=(10,5))
    plt.scatter(df_caps["mid_sec"], df_caps["text_length"], s=8, alpha=0.4)
    plt.xlabel("Video Time (sec)")
    plt.ylabel("Caption Length (chars)")
    plt.title("Caption Length vs Video Time")
    figsave(os.path.join(EDA_DIR, "caption_length_vs_time_scatter.png"))

# ----------------------------
# CLEAN & EXPORT (for future labs)
# ----------------------------
def export_clean_versions(comments_df, captions_sent_df):
    # Cleaned tokens
    comments_df["cleaned_tokens"] = comments_df["text"].astype(str).apply(lambda x: clean_pipeline(x, use_lemmatization=True))
    captions_sent_df["cleaned_tokens"] = captions_sent_df["caption_sentence"].astype(str).apply(lambda x: clean_pipeline(x, use_lemmatization=True))

    # Save CSV
    comments_df.to_csv(os.path.join(CLEAN_DIR, "cleaned_comments.csv"), index=False, encoding="utf-8-sig")
    captions_sent_df.to_csv(os.path.join(CLEAN_DIR, "cleaned_captions.csv"), index=False, encoding="utf-8-sig")

# ----------------------------
# MAIN
# ----------------------------
def main():
    # 3) Profiling previews (helps you adjust parsers if needed)
    profile_report = []
    if is_file(COMMENTS_TXT):
        profile_report.append(profile_file_quick(COMMENTS_TXT))
    if is_file(CAPTIONS_VTT):
        profile_report.append(profile_file_quick(CAPTIONS_VTT))
    if profile_report:
        save_txt(os.path.join(EDA_DIR, "profile_previews.txt"), "\n\n".join(profile_report))

    # 5A) Comments: prefer CSV; also parse comments.txt if present (for comparison)
    comments_df = load_comments_csv(COMPONENTS_CSV if (COMPONENTS_CSV := COMMENTS_CSV) else COMMENTS_CSV)
    # If you want to also build a txt-structured DataFrame:
    txt_df = structure_comments_from_txt(COMMENTS_TXT)
    if not txt_df.empty:
        txt_df.to_csv(os.path.join(TABLES_DIR, "comments_from_txt_blocks.csv"), index=False, encoding="utf-8-sig")

    # 5B) Captions
    caps_df = structure_captions_from_vtt(CAPTIONS_VTT)
    caps_df.to_csv(os.path.join(TABLES_DIR, "captions_raw_rows.csv"), index=False, encoding="utf-8-sig")
    # sentence list for downstream analysis
    caps_sent_df = captions_to_sentences(caps_df)
    caps_sent_df.to_csv(os.path.join(TABLES_DIR, "captions_sentences.csv"), index=False, encoding="utf-8-sig")

    # 5C) Core EDA visuals
    if not caps_df.empty:
        plot_hist_lengths_captions_vs_comments(caps_df, comments_df)
        plot_ttr_bars(caps_df, comments_df)
        plot_top_words(caps_df, comments_df, n=20)
    else:
        # Comments-only case
        plot_ttr_bars(pd.DataFrame({"text": [""]}), comments_df)
        plot_top_words(pd.DataFrame({"text": [""]}), comments_df, n=20)

    plot_comment_length_vs_likes(comments_df)
    plot_comments_over_time(comments_df)
    plot_caption_length_vs_time(caps_df)

    # 6) Export cleaned data for future labs
    export_clean_versions(comments_df, caps_sent_df)

    # 7) Checklist saved
    checklist = [
        "Profiled files and noted structure ✅",
        "Parsed comments (CSV and optional TXT blocks) ✅",
        "Parsed captions (.vtt) and built sentence list ✅",
        "Normalized, tokenized, stopword-removed, lemmatized ✅",
        "Saved cleaned data: outputs/CLEAN/cleaned_comments.csv, cleaned_captions.csv ✅",
        "Generated EDA visuals in outputs/EDA at 300 DPI ✅",
    ]
    save_txt(os.path.join(OUT_DIR, "CHECKLIST.txt"), "\n".join(checklist))

    print(f"Done. See:\n- {EDA_DIR}\n- {CLEAN_DIR}\n- {TABLES_DIR}")

if __name__ == "__main__":
    main()
