import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Setup
# ------------------------------
csv_file = "comments_all.csv"
eda_dir = "EDA"
os.makedirs(eda_dir, exist_ok=True)

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv(csv_file)

# Ensure proper types
df["text"] = df["text"].astype(str)
df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)

# Convert published (Unix timestamp) -> datetime
df["published_dt"] = pd.to_datetime(df["published"], unit="s", errors="coerce")

print(f"✅ Loaded {len(df)} comments")

# ------------------------------
# Save dataset overview
# ------------------------------
with open(os.path.join(eda_dir, "basic_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"Total comments: {len(df)}\n\n")
    f.write("Columns & unique counts:\n")
    for col in df.columns:
        f.write(f"- {col}: {df[col].nunique()} unique\n")

    f.write("\nMissing values:\n")
    f.write(str(df.isna().sum()))
    f.write("\n\nSample rows:\n")
    f.write(str(df.sample(5)))

# ------------------------------
# Plot 1: Distribution of Likes
# ------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["like_count"], bins=50, log_scale=(False, True))
plt.title("Distribution of Like Counts")
plt.xlabel("Likes")
plt.ylabel("Frequency (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, "likes_distribution.png"))
plt.close()

# ------------------------------
# Plot 2: Comment Length Distribution
# ------------------------------
df["text_length"] = df["text"].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(df["text_length"], bins=60, kde=True)
plt.title("Distribution of Comment Lengths")
plt.xlabel("Characters per Comment")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, "comment_length_distribution.png"))
plt.close()

# ------------------------------
# Plot 3: Top Authors
# ------------------------------
top_authors = df["author"].value_counts().head(20)

plt.figure(figsize=(10,6))
sns.barplot(x=top_authors.values, y=top_authors.index, palette="viridis")
plt.title("Top 20 Authors by Comment Count")
plt.xlabel("Number of Comments")
plt.ylabel("Author")
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, "top_authors.png"))
plt.close()

# ------------------------------
# Plot 4: Timeline of Comments
# ------------------------------
if not df["published_dt"].isna().all():
    daily_counts = df.groupby(df["published_dt"].dt.date).size()

    plt.figure(figsize=(12,6))
    daily_counts.plot(kind="line", marker="o")
    plt.title("Number of Comments Over Time")
    plt.xlabel("Date")
    plt.ylabel("Comments per Day")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "comments_over_time.png"))
    plt.close()

# ------------------------------
# Plot 5: Top-level vs Replies
# ------------------------------
df["is_reply"] = df["parent_id"].apply(lambda x: "Reply" if x != "root" else "Top-level")

plt.figure(figsize=(6,5))
sns.countplot(x="is_reply", data=df, palette="pastel")
plt.title("Top-level Comments vs Replies")
plt.xlabel("Comment Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, "top_vs_replies.png"))
plt.close()

# ------------------------------
# Plot 6: Word Cloud (optional)
# ------------------------------
try:
    from wordcloud import WordCloud

    all_text = " ".join(df["text"].tolist())
    wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)

    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Comments", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "wordcloud.png"))
    plt.close()
except ImportError:
    print("⚠️ wordcloud not installed. Run: pip install wordcloud")

print(f"🎯 EDA complete. All outputs saved in '{eda_dir}'")