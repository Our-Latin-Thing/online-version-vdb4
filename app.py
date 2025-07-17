import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)

logger.info("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger.info("Initializing Pinecone client...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(
    os.getenv("PINECONE_INDEX_NAME"),
    pool_threads=50
)
logger.info("Clients initialized successfully")

@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({"message": "API is working!"})

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embed = response.data[0].embedding

    result = index.query(vector=embed, top_k=8, include_metadata=True)

    matches = []
    for match in result['matches']:
        meta = match['metadata']
        matches.append({
            "title": meta.get("title", ""),
            "description": meta.get("text", ""),
            "tags": meta.get("tags", []),
            "address": meta.get("address", ""),
            "image_url": meta.get("image_url", ""),
            "score": match.get("score", 0)
        })

    return jsonify({"results": matches})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)