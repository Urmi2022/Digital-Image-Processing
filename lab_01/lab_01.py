import json, csv

# 1) Point to your .info.json file
json_file = r"This is the Apple iPhone 16 [0iIPe9XrpcM].info.json"

# 2) Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3) Locate comments
comments = []
if "comments" in data and isinstance(data["comments"], list):
    comments = data["comments"]
elif "entries" in data and data["entries"]:
    comments = data["entries"][0].get("comments", [])

if not comments:
    raise RuntimeError("No comments found in JSON. Open the JSON file and search for 'comments' to confirm.")

# 4) Normalize and write to CSV
out_file = "comments_all.csv"
with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "comment_id", "parent_id", "author", "text",
        "published", "like_count", "is_top_level"
    ])
    writer.writeheader()

    for c in comments:
        writer.writerow({
            "comment_id": c.get("id") or c.get("commentId") or "",
            "parent_id": c.get("parent") or c.get("parentId") or "",
            "author": c.get("author") or c.get("authorText") or "",
            "text": (c.get("text") or c.get("textHtml") or c.get("textOriginal") or "").replace("\n"," ").strip(),
            "published": c.get("timestamp") or c.get("publishedTimeText") or c.get("published") or "",
            "like_count": c.get("like_count") or c.get("likes") or 0,
            "is_top_level": "" if c.get("parent") or c.get("parentId") else "TRUE",
        })

print(f"✅ Extracted {len(comments)} comments to {out_file}")