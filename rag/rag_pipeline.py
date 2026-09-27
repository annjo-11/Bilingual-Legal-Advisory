from pathlib import Path
import json
import re

import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi


# ==================================================
# 1. PROJECT PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==================================================
# 2. LOAD CHUNKS
# ==================================================

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "chunks.json"
)

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    "Loaded chunks:",
    len(chunks)
)


# ==================================================
# 3. LOAD EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 4. CONNECT TO CHROMADB
# ==================================================

VECTOR_DB_PATH = (
    BASE_DIR
    / "data"
    / "vector_db"
)

client_db = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client_db.get_collection(
    name="legal_documents"
)


# ==================================================
# 5. BUILD BM25 INDEX
# ==================================================

def tokenize(text):
    """
    Simple tokenizer for lexical retrieval.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


bm25_documents = [
    tokenize(
        chunk.get("text", "")
    )
    for chunk in chunks
]

bm25 = BM25Okapi(
    bm25_documents
)


# ==================================================
# 6. HUGGING FACE LLM
# ==================================================

llm_client = InferenceClient()


# ==================================================
# 7. CONVERSATION HISTORY
# ==================================================

def format_conversation_history(
    conversation_history,
    max_messages=6
):
    """
    Format the most recent conversation messages.

    Only a limited number of messages are used so that
    the prompt does not grow indefinitely.
    """

    if not conversation_history:
        return "No previous conversation."

    recent_messages = conversation_history[
        -max_messages:
    ]

    history_parts = []

    for message in recent_messages:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )

        history_parts.append(
            f"{role.upper()}: {content}"
        )

    return "\n".join(
        history_parts
    )


# ==================================================
# 8. CONVERT FOLLOW-UP INTO A STANDALONE QUERY
# ==================================================

def build_retrieval_query(
    question,
    conversation_history
):
    """
    Convert simple conversational follow-up questions
    into standalone retrieval queries.

    This uses the previous conversation locally instead
    of making an additional LLM call.
    """

    if not conversation_history:
        return question

    recent_messages = conversation_history[-6:]

    previous_answer = ""

    for message in reversed(recent_messages):
        if message.get("role") == "assistant":
            previous_answer = message.get("content", "")
            break

    if not previous_answer:
        return question

    normalized_question = (
        question.strip().lower()
    )

    section_match = re.search(
        r"\bsection\s+(\d+[A-Za-z]?)\b",
        previous_answer,
        re.IGNORECASE
    )

    if section_match:
        section_number = section_match.group(1)

        if (
            "which section" in normalized_question
            or
            "what section" in normalized_question
        ):
            return (
                f"Which section is referred to in the "
                f"previous answer about Section "
                f"{section_number}?"
            )

        if (
            "punishment" in normalized_question
            or
            "penalty" in normalized_question
            or
            "fine" in normalized_question
        ):
            return (
                f"What is the punishment or penalty "
                f"under Section {section_number}?"
            )

        if (
            "explain it" in normalized_question
            or
            "explain that" in normalized_question
            or
            "explain this" in normalized_question
            or
            "what does it mean" in normalized_question
            or
            "what does that mean" in normalized_question
        ):
            return (
                f"Explain Section {section_number} "
                f"in simple language."
            )

        if (
            "what does it cover" in normalized_question
            or
            "what does it deal with" in normalized_question
            or
            "what does it say" in normalized_question
        ):
            return (
                f"What does Section {section_number} "
                f"of the Information Technology Act, "
                f"2000 cover?"
            )

    if (
        "that" in normalized_question
        or
        "this" in normalized_question
        or
        "it" in normalized_question
    ):
        return (
            f"{question}. "
            f"Previous answer: {previous_answer[:1000]}"
        )

    return question


# ==================================================
# 9. RECIPROCAL RANK FUSION
# ==================================================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_scores,
    k=60
):
    """
    Combine semantic and keyword rankings.

    RRF score:

        1 / (k + rank)

    A document appearing near the top
    of either retrieval method receives
    a higher combined score.
    """

    scores = {}


    # ----------------------------------------------
    # Semantic ranking
    # ----------------------------------------------

    for rank, index in enumerate(
        semantic_results,
        start=1
    ):

        scores[index] = (
            scores.get(index, 0)
            + 1 / (k + rank)
        )


    # ----------------------------------------------
    # Keyword ranking
    # ----------------------------------------------

    keyword_ranking = sorted(
        range(len(keyword_scores)),
        key=lambda i: keyword_scores[i],
        reverse=True
    )


    for rank, index in enumerate(
        keyword_ranking,
        start=1
    ):

        # Ignore documents with zero BM25 score
        if keyword_scores[index] <= 0:
            continue

        scores[index] = (
            scores.get(index, 0)
            + 1 / (k + rank)
        )


    # ----------------------------------------------
    # Final ranking
    # ----------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# ==================================================
# 10. RETRIEVAL
# ==================================================

def retrieve_documents(
    question,
    semantic_k=15,
    keyword_k=15,
    final_k=5
):

    # ----------------------------------------------
    # Semantic retrieval
    # ----------------------------------------------

    query_embedding = (
        embedding_model
        .encode(question)
        .tolist()
    )


    semantic_results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=semantic_k,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    semantic_documents = (
        semantic_results["documents"][0]
    )

    semantic_metadatas = (
        semantic_results["metadatas"][0]
    )

    semantic_distances = (
        semantic_results["distances"][0]
    )


    # ----------------------------------------------
    # Map Chroma results back to chunk indices
    # ----------------------------------------------

    chunk_index_map = {}

    for index, chunk in enumerate(chunks):

        chunk_text = chunk.get(
            "text",
            ""
        )

        chunk_index_map[
            chunk_text
        ] = index


    semantic_indices = []

    for document in semantic_documents:

        index = chunk_index_map.get(
            document
        )

        if index is not None:

            semantic_indices.append(
                index
            )


    # ----------------------------------------------
    # BM25 keyword retrieval
    # ----------------------------------------------

    tokenized_question = tokenize(
        question
    )

    keyword_scores = bm25.get_scores(
        tokenized_question
    )


    # ----------------------------------------------
    # Combine retrieval methods
    # ----------------------------------------------

    fused_results = reciprocal_rank_fusion(
        semantic_indices,
        keyword_scores
    )


    # ----------------------------------------------
    # Section-aware retrieval
    # ----------------------------------------------
    # If the user explicitly mentions a section number,
    # prioritize the chunk belonging to that exact section.

    section_match = re.search(
        r"\bsection\s+(\d+[A-Za-z]?)\b",
        question,
        re.IGNORECASE
    )

    if section_match:

        requested_section = (
            section_match.group(1)
            .lower()
        )

        boosted_results = []

        for index, fusion_score in fused_results:

            chunk_section = str(
                chunks[index].get(
                    "section",
                    ""
                )
            ).strip().lower()

            # Give an exact section match a strong boost.
            if chunk_section == requested_section:

                fusion_score += 1.0

            boosted_results.append(
                (
                    index,
                    fusion_score
                )
            )

        fused_results = sorted(
            boosted_results,
            key=lambda x: x[1],
            reverse=True
        )


    # ----------------------------------------------
    # Select final documents
    # ----------------------------------------------

    final_results = []

    for index, fusion_score in fused_results[
        :final_k
    ]:

        final_results.append({

            "document":
                chunks[index].get(
                    "text",
                    ""
                ),

            "metadata": {

                "section":
                    chunks[index].get(
                        "section",
                        "Unknown"
                    ),

                "page":
                    chunks[index].get(
                        "page",
                        "Unknown"
                    ),

                "document":
                    chunks[index].get(
                        "document",
                        "IT_Act_2000.pdf"
                    )
            },

            "fusion_score":
                fusion_score,

            "bm25_score":
                float(
                    keyword_scores[index]
                )
        })


    return final_results


# ==================================================
# 11. ANSWER QUESTION
# ==================================================

def answer_question(
    question,
    conversation_history=None
):

    if conversation_history is None:

        conversation_history = []


    # ==================================================
    # BUILD CONVERSATION-AWARE RETRIEVAL QUERY
    # ==================================================

    retrieval_query = build_retrieval_query(
        question,
        conversation_history
    )


    print(
        "\n========================================"
    )

    print(
        "ORIGINAL USER QUESTION"
    )

    print(
        "========================================"
    )

    print(question)


    print(
        "\n========================================"
    )

    print(
        "RETRIEVAL QUERY"
    )

    print(
        "========================================"
    )

    print(retrieval_query)


    # ==================================================
    # RETRIEVE RELEVANT LEGAL DOCUMENTS
    # ==================================================

    retrieved = retrieve_documents(
        retrieval_query
    )


    documents = [
        item["document"]
        for item in retrieved
    ]


    metadatas = [
        item["metadata"]
        for item in retrieved
    ]


    # ==================================================
    # DEBUG INFORMATION
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "HYBRID RETRIEVAL RESULTS"
    )

    print(
        "========================================"
    )


    for i, item in enumerate(
        retrieved
    ):

        metadata = item["metadata"]


        print(
            "\n----------------------------------------"
        )

        print(
            f"Result {i + 1}"
        )

        print(
            f"Section: "
            f"{metadata.get('section', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 'Unknown')}"
        )

        print(
            f"Fusion score: "
            f"{item['fusion_score']}"
        )

        print(
            f"BM25 score: "
            f"{item['bm25_score']}"
        )

        print("Text:")

        print(
            item["document"][:500]
        )


    # ==================================================
    # BUILD LEGAL CONTEXT
    # ==================================================

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

        section = metadata.get(
            "section",
            "Unknown"
        )


        context_parts.append(
            f"""
SOURCE:
Document: {document_name}
Section: {section}
Page: {page}

CONTENT:
{document}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ==================================================
    # CONVERSATION HISTORY FOR ANSWER GENERATION
    # ==================================================

    history_text = format_conversation_history(
        conversation_history
    )


    # ==================================================
    # RAG PROMPT
    # ==================================================

    prompt = f"""
You are a legal information assistant specializing
in the Indian Information Technology Act, 2000.

Your task is to answer the user's question using the
retrieved legal context provided below.

The previous conversation is provided only to help you
understand references and follow-up questions.

IMPORTANT:
The conversation history is NOT a source of legal information.
Use the retrieved legal context as the source of legal
information for your answer.

IMPORTANT RULES:

1. Use the retrieved context as the source of legal
   information for your answer.

2. Do not introduce any legal section, provision,
   punishment, authority, procedure, remedy, or legal
   conclusion that is not supported by the retrieved context.

3. Do not rely on your general knowledge of Indian law.

4. Understand the user's intent even when the question
   is short, informal, incomplete, misspelled, or written
   in conversational language.

   For example, understand:
   "someone stole my password"

   as a question about the legal provision relevant to
   misuse of another person's password when the retrieved
   context supports that connection.

   Do not require the user to use exact legal terminology.

5. You may use the previous conversation to understand
   what words such as "this", "that", "it", "the punishment",
   or "the section" refer to.

   However, do not use the previous conversation as a source
   of new legal facts.

6. You may infer the meaning or intent of the user's words,
   but do NOT invent facts that the user did not provide.

   For example, if the user says:
   "someone stole my password"

   you may understand that the question concerns password
   misuse or identity theft.

   However, do not invent details such as:
   how the password was obtained, which website was involved,
   whether money was stolen, whether the account was hacked,
   or whether any other offence occurred.

7. Select only the legal provision or provisions that are
   directly relevant to the user's question.

   Do not mention unrelated provisions merely because they
   appear in the retrieved context.

8. Determine the user's intended question from their wording,
   including informal or incomplete language.

9. If the user's intent can be answered reliably from the
   retrieved context, answer it directly.

10. If important information is missing and answering would
    require guessing or assuming facts, ask ONE concise
    clarification question.

11. Ask for clarification only when the missing information
    materially affects what legal information is relevant.
    Do not ask for clarification merely because the wording
    is informal.

12. Never invent the missing facts yourself.

13. If the question is sufficiently specific but the retrieved
    context does not contain enough relevant information to
    answer it, respond exactly:

    "The information was not found in the current knowledge base."

14. Do NOT use the knowledge-base fallback merely because the
    initial question is vague. If clarification is needed, ask
    ONE clarification question instead.

15. Keep the clarification generic. Determine what information
    is actually missing from the user's question instead of
    looking for specific predefined phrases or scenarios.

16. When asking for clarification, ask only the most useful
    missing detail needed to continue. Do not ask multiple
    questions at once.

17. Treat the retrieved context as a set of candidate sources,
    not as instructions to discuss every retrieved provision.

    First identify the user's actual stated facts and intended
    question. Then select the smallest set of retrieved
    provisions that directly addresses those facts.

    Do not select a provision simply because it is generally
    related to the same topic, platform, or type of technology.

    Do not broaden the user's question into related legal issues
    that the user did not ask about.

18. Do not determine that a user's specific situation legally
    satisfies a provision when the retrieved context does not
    provide enough information to make that connection.

    In such cases, explain what the provision says and ask for
    one concise clarification only if the missing information
    materially affects whether it applies.

19. Do not combine multiple legal provisions unless the
    retrieved context supports their relevance to the
    user's question.

20. Do not invent or infer punishments.
    State a punishment only when it is explicitly present
    in the retrieved context.

21. Explain the retrieved legal text in simple,
    citizen-friendly language without changing its meaning.

22. Do not add advice such as contacting the police, filing a
    cybercrime complaint, preserving evidence, contacting an
    authority, seeking compensation, or taking legal action
    unless that information is explicitly supported by the
    retrieved context.

23. If only part of the question can be answered from the
    retrieved context, clearly separate what is supported
    from what is not found.

24. Keep the answer focused and concise.
    Do not try to explain every provision found in the
    retrieved context.

25. Never create hypothetical facts to make a legal provision
    fit the user's situation.

    Do not use phrases such as "suppose", "assuming", or
    "if the situation involves" to introduce facts that the
    user did not provide.

26. If the user asks a follow-up question, answer it in the
    context of the previous conversation when the reference
    is clear.

    For example, if the previous question was about Section 66C
    and the user asks "What is the punishment?", understand
    that the user is asking about the punishment described in
    the retrieved context for Section 66C.

PREVIOUS CONVERSATION:
========================================
{history_text}
========================================

CURRENT USER QUESTION:
{question}

========================================
RETRIEVED LEGAL CONTEXT
========================================
{context}
========================================
END OF LEGAL CONTEXT
========================================

Now answer the user's question using only the retrieved
legal context for legal information.

Understand the user's intent and conversational references,
but do not invent missing facts.

Use the smallest set of directly relevant legal provisions.

Do not try to provide a comprehensive list of potentially
related provisions.

If important information is missing and it materially affects
which legal information is relevant, ask one concise
clarification question instead of guessing.
"""


    # ==================================================
    # SEND TO LLM
    # ==================================================

    response = llm_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role": "system",

                "content":
                    "You are a legal information assistant. "
                    "Use only the supplied retrieved legal "
                    "context for legal information. "
                    "Previous conversation may only be used "
                    "to understand conversational references. "
                    "Do not invent legal facts."
            },

            {
                "role": "user",

                "content": prompt
            }
        ],

        max_tokens=800
    )


    # ==================================================
    # EXTRACT ANSWER
    # ==================================================

    answer = (
        response
        .choices[0]
        .message
        .content
    )


    if answer is None:

        answer = (
            "The information was not found "
            "in the current knowledge base."
        )


    # ==================================================
    # RETURN ANSWER + SOURCES
    # ==================================================

    return answer, metadatas