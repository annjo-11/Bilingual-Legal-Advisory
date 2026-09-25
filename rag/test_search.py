from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_PATH = BASE_DIR / "data" / "vector_db"


# ==================================================
# EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_collection(
    name="legal_documents"
)


# ==================================================
# QUESTION
# ==================================================

question = "what is identity theft under section 66C"


# ==================================================
# EMBEDDING
# ==================================================

query_embedding = embedding_model.encode(
    question
).tolist()


# ==================================================
# SEARCH
# ==================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# ==================================================
# DISPLAY
# ==================================================

print("\n========================================")
print("QUESTION")
print("========================================")

print(question)


print("\n========================================")
print("RESULTS")
print("========================================")


for i in range(len(results["documents"][0])):

    print("\n----------------------------------------")

    print(
        "RESULT:",
        i + 1
    )

    print(
        "SECTION:",
        results["metadatas"][0][i].get(
            "section",
            "Unknown"
        )
    )

    print(
        "PAGE:",
        results["metadatas"][0][i].get(
            "page",
            "Unknown"
        )
    )

    print(
        "DISTANCE:",
        results["distances"][0][i]
    )

    print("\nTEXT:")

    print(
        results["documents"][0][i]
    )