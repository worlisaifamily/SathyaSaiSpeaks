from flask import Flask, request, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

app = Flask(__name__)

# Load CSV and model
df = pd.read_csv("shorts.csv", encoding='utf-8-sig')
model = SentenceTransformer('all-MiniLM-L6-v2')
title_embeddings = model.encode(df['title'].tolist(), convert_to_tensor=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        
        if "search" in request.form:
+            query = request.form["query"]
            query_embedding = model.encode([query], convert_to_tensor=True)
            similarities = cosine_similarity(query_embedding, title_embeddings)[0]
            best_idx = similarities.argmax()
            result = {
                "title": df.iloc[best_idx]["title"],
                "link": df.iloc[best_idx]["link"],
                "score": f"{similarities[best_idx]:.2f}",
                "mode": "search"
            }

        elif "random" in request.form:
            random_row = df.sample(1).iloc[0]
            result = {
                "title": random_row["title"],
                "link": random_row["link"],
                "score": None,
                "mode": "random"
            }

    return render_template("index.html", result=result)

# ✅ Proper block to support both local and Render deployment
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
