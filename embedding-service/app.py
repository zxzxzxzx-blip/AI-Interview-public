import hashlib
import math
import os
import re
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
VECTOR_SIZE = 768

app = FastAPI()

# The initial model download may be unavailable on restricted networks. Keep the
# service available with deterministic 768-dimensional vectors in that case.
try:
    model = SentenceTransformer(MODEL_NAME)
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)
    print(f"Embedding model unavailable; using local fallback vectors: {model_error}")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    return {
        "status": "UP",
        "model": MODEL_NAME,
        "mode": "sentence-transformer" if model else "hash-fallback",
    }


def fallback_embedding(text: str) -> List[float]:
    """Generate stable lexical vectors when the external model is unavailable."""
    values = [0.0] * VECTOR_SIZE
    terms = re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) or [text]

    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            bucket = int.from_bytes(digest[offset : offset + 4], "big")
            values[bucket % VECTOR_SIZE] += 1.0 if bucket & 1 else -1.0

    if NORMALIZE:
        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude:
            values = [value / magnitude for value in values]
    return values


@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    if model is None:
        return {"embeddings": [fallback_embedding(text) for text in request.texts]}

    vectors = model.encode(
        request.texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=False,
    )
    return {"embeddings": vectors.tolist()}
