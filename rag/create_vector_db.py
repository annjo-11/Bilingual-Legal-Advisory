from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. PROJECT PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
VECTOR_DB_PATH = BASE_DIR / "data" / "vector_db"


# ==================================================
# 2. LOAD CHUNKS
# ==================================================

print("Loading chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Number of chunks:", len(chunks))


# ==================================================
# 3. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 4. CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)


# ==================================================
# 5. DELETE OLD COLLECTION
# ==================================================

print("Removing old collection...")

try:
    client.delete_collection(
        name="legal_documents"
    )

    print("Old collection deleted.")

except Exception:
    print("No old collection found.")


# ==================================================
# 6. CREATE NEW COLLECTION
# ==================================================

collection = client.create_collection(
    name="legal_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ==================================================
# 7. PREPARE DATA
# ==================================================

documents = []
metadatas = []
ids = []


for i, chunk in enumerate(chunks):

    # ----------------------------------------------
    # Get text
    # ----------------------------------------------

    if "text" in chunk:
        text = chunk["text"]

    elif "content" in chunk:
        text = chunk["content"]

    elif "chunk" in chunk:
        text = chunk["chunk"]

    else:
        print("Could not find text in chunk:", i)
        continue


    # ----------------------------------------------
    # Metadata
    # ----------------------------------------------

    section = chunk.get(
        "section",
        "Unknown"
    )

    page = chunk.get(
        "page",
        "Unknown"
    )

    document_name = chunk.get(
        "document",
        "IT_Act_2000.pdf"
    )


    documents.append(text)

    metadatas.append({
        "section": str(section),
        "page": str(page),
        "document": str(document_name)
    })

    ids.append(
        f"chunk_{i}"
    )


# ==================================================
# 8. CREATE EMBEDDINGS
# ==================================================

print("Creating embeddings...")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
).tolist()


# ==================================================
# 9. STORE IN CHROMADB
# ==================================================

print("Adding documents to ChromaDB...")

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# ==================================================
# 10. VERIFY
# ==================================================

print("\n========================================")
print("VECTOR DATABASE CREATED")
print("========================================")

print(
    "Documents stored:",
    collection.count()
)

print(
    "Collection:",
    collection.name
)

print(
    "Database path:",
    VECTOR_DB_PATH
)

print("========================================")