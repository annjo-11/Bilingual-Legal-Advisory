from rag_pipeline import retrieve_documents


# ==================================================
# QUESTION
# ==================================================

question = "what is identity theft under section 66C"


# ==================================================
# RETRIEVE
# ==================================================

results = retrieve_documents(
    question,
    semantic_k=15,
    keyword_k=15,
    final_k=5
)


# ==================================================
# DISPLAY
# ==================================================

print("\n========================================")
print("QUESTION")
print("========================================")

print(question)


print("\n========================================")
print("HYBRID RETRIEVAL RESULTS")
print("========================================")


for i, result in enumerate(results):

    metadata = result["metadata"]

    print("\n----------------------------------------")
    print(f"RESULT: {i + 1}")

    print(
        "SECTION:",
        metadata.get("section", "Unknown")
    )

    print(
        "PAGE:",
        metadata.get("page", "Unknown")
    )

    print(
        "FUSION SCORE:",
        result["fusion_score"]
    )

    print(
        "BM25 SCORE:",
        result["bm25_score"]
    )

    print("\nTEXT:")

    print(result["document"])