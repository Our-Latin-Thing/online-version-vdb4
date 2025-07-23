from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pinecone.grpc import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

openai = OpenAI(api_key=openai_api_key)

pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index(pinecone_index_name)

app = Flask(__name__)
CORS(app)

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Embed the query using OpenAI
    embedding_response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embedding = embedding_response.data[0].embedding

    # Query Pinecone index
    pinecone_response = index.query(
        vector=embedding,
        top_k=8,
        include_metadata=True
    )

    matches = []
    for match in pinecone_response.matches:
        meta = match.metadata or {}
        matches.append({
            "title": meta.get("title", ""),
            "description": meta.get("text", ""),
            "tags": meta.get("tags", []),
            "address": meta.get("address", ""),
            "image_url": meta.get("image_url", ""),
            "url": meta.get("url", ""),  # ✅ Ensure frontend receives the URL
            "score": match.score
        })

    return jsonify({"results": matches})
