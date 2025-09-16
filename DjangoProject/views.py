from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from django.conf import settings
import numpy as np

# Path to dataset
csv_path = os.path.join(settings.BASE_DIR, "myapp", "dataset", "formatted_jobs.csv")

# Load dataset (drop ID_num if exists)
df = pd.read_csv(csv_path)
if "ID_num" in df.columns:
    df = df.drop(columns=["ID_num"])

# Create UserInfo column (skills + description for comparison)
df["UserInfo"] = (
    df["Short_description"].fillna('') + " " +
    df["Skills_required"].fillna('')
)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute embeddings
job_embeddings = model.encode(df["UserInfo"].tolist(), normalize_embeddings=True)

def recommend_jobs(user_input, top_k=5):
    # Encode user input
    user_embedding = model.encode([user_input], normalize_embeddings=True)

    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding, job_embeddings)[0]

    # Top K results
    top_idx = np.argsort(similarities)[::-1][:top_k]

    # Return job_title, Industry, Pay_grade
    return df.iloc[top_idx][["job_title", "Industry", "Pay_grade"]].to_dict(orient="records")


# Chat UI
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def recommender_view(request):
    if request.method == "POST":
        job_desc = request.POST.get("job_desc", "")
        skills = request.POST.get("skills", "")

        if not job_desc.strip() or not skills.strip():
            return JsonResponse({"error": "Please provide both job description and skills."})

        # Concat job_desc + skills
        user_input = f"{job_desc}. Skills: {skills}"

        # Get recommendations
        results = recommend_jobs(user_input, top_k=3)

        return JsonResponse({"recommendations": results})

    return render(request, "chat.html")

