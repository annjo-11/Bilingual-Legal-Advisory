import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

VECTOR_DB_FOLDER = BASE_DIR / "data" / "vector_db"


# --------------------------------------------------
# CREATE VECTOR DATABASE FOLDER
# --------------------------------------------------

VECTOR_DB_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

print("Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# CREATE CHROMADB CLIENT
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_FOLDER)
)


# --------------------------------------------------
# CREATE COLLECTION
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="legal_documents"
)


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

ids = []
documents = []
metadatas = []


for chunk in chunks:

    ids.append(chunk["chunk_id"])

    documents.append(chunk["text"])

    metadatas.append({
        "document": chunk["document"],
        "page": chunk["page"]
    })


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

print("\nCreating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)


# --------------------------------------------------
# STORE EVERYTHING IN CHROMADB
# --------------------------------------------------

print("\nStoring data in vector database...")

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\nVector database created successfully!")

print(
    f"Total documents stored: {collection.count()}"
)

print(
    f"Database location: {VECTOR_DB_FOLDER}"
)