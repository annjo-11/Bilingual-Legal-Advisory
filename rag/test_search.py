from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_FOLDER = BASE_DIR / "data" / "vector_db"


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# CONNECT TO VECTOR DATABASE
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_FOLDER)
)

collection = client.get_collection(
    name="legal_documents"
)


# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

query = "What is identity theft under the Information Technology Act?"


# --------------------------------------------------
# CREATE QUERY EMBEDDING
# --------------------------------------------------

query_embedding = model.encode(query).tolist()


# --------------------------------------------------
# SEARCH VECTOR DATABASE
# --------------------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nQUERY:")
print(query)

print("\nMOST RELEVANT RESULTS:\n")


for i in range(len(results["documents"][0])):

    document = results["metadatas"][0][i]
    text = results["documents"][0][i]

    print("=" * 70)

    print(
        f"Result {i + 1}"
    )

    print(
        f"Document: {document['document']}"
    )

    print(
        f"Page: {document['page']}"
    )

    print("\nText:")

    print(text)

    print()