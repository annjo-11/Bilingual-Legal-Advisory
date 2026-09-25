from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# ==================================================
# 1. PROJECT PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==================================================
# 2. LOAD EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 3. CONNECT TO CHROMADB
# ==================================================

VECTOR_DB_PATH = BASE_DIR / "data" / "vector_db"

client_db = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client_db.get_collection(
    name="legal_documents"
)


# ==================================================
# 4. HUGGING FACE LLM
# ==================================================

llm_client = InferenceClient()


# ==================================================
# 5. RAG FUNCTION
# ==================================================

def answer_question(question):

    # ------------------------------------------------
    # Convert question into embedding
    # ------------------------------------------------

    query_embedding = embedding_model.encode(
        question
    ).tolist()


    # ------------------------------------------------
    # Search ChromaDB
    # ------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    # ------------------------------------------------
    # Get results
    # ------------------------------------------------

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]


    # ------------------------------------------------
    # DEBUG INFORMATION
    # ------------------------------------------------

    print("\n========================================")
    print("USER QUESTION")
    print("========================================")

    print(question)


    print("\n========================================")
    print("RETRIEVED RESULTS")
    print("========================================")


    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances)
    ):

        print("\n----------------------------------------")

        print(f"Result {i + 1}")

        print(
            f"Section: {metadata.get('section', 'Unknown')}"
            )

        print(
            f"Page: {metadata.get('page', 'Unknown')}"
        )

        print(
            f"Distance: {distance}"
        )

        print("Text:")

        print(document[:500])


    # ------------------------------------------------
    # BUILD CONTEXT
    # ------------------------------------------------

    context_parts = []


    for document, metadata in zip(
        documents,
        metadatas
    ):

        page = metadata.get(
            "page",
            "Unknown"
        )

        document_name = metadata.get(
            "document",
            "IT_Act_2000.pdf"
        )


        context_parts.append(
            f"""
SOURCE:
Document: {document_name}
Page: {page}

CONTENT:
{document}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ------------------------------------------------
    # RAG PROMPT
    # ------------------------------------------------

    prompt = f"""
You are a legal information assistant specializing
in the Indian Information Technology Act, 2000.

Your task is to answer the user's question using
the retrieved legal context provided below.

IMPORTANT RULES:

1. Use the retrieved context as the primary source.
2. If the answer is present in the context, answer it.
3. Do not say that the information is unavailable
   merely because the exact wording of the question
   is different from the wording in the document.
4. You may explain the retrieved legal text in simple
   language.
5. Do not invent legal sections, punishments or facts.
6. If the retrieved context genuinely does not contain
   information that answers the question, say exactly:

"The information was not found in the current knowledge base."

USER QUESTION:
{question}

========================================
RETRIEVED LEGAL CONTEXT
========================================

{context}

========================================
END OF LEGAL CONTEXT
========================================

Now answer the user's question.
"""


    # ------------------------------------------------
    # SEND TO LLM
    # ------------------------------------------------

    response = llm_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content":
                    "You answer questions using the supplied "
                    "legal context and do not invent legal facts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=800
    )


    # ------------------------------------------------
    # EXTRACT ANSWER
    # ------------------------------------------------

    answer = response.choices[0].message.content

    if answer is None:
        answer = "The information was not found in the current knowledge base."


    # ------------------------------------------------
    # RETURN ANSWER + SOURCES
    # ------------------------------------------------

    return answer, metadatas