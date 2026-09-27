from rag.rag_pipeline import (
    build_retrieval_query,
    retrieve_documents
)

conversation = []

questions = [
    "Someone stole my password and used my account.",
    "What is the punishment?",
    "Which section is that?",
    "Can you explain it simply?"
]

for question in questions:

    print("\n" + "=" * 60)
    print("USER:")
    print(question)

    retrieval_query = build_retrieval_query(
        question,
        conversation
    )

    print("\nRETRIEVAL QUERY:")
    print(retrieval_query)

    retrieved = retrieve_documents(
        retrieval_query
    )

    print("\nTOP RETRIEVED RESULTS:")

    for i, item in enumerate(retrieved):

        metadata = item["metadata"]

        print(
            f"{i + 1}. "
            f"Section {metadata.get('section')} "
            f"| Page {metadata.get('page')} "
            f"| Score {item['fusion_score']}"
        )

    # Simulate the previous assistant response
    # so that the next question has conversation context.

    if question == questions[0]:

        answer = (
            "Section 66C deals with fraudulent or dishonest "
            "use of another person's password or unique "
            "identification feature."
        )

    elif question == questions[1]:

        answer = (
            "Under Section 66C, punishment may extend to "
            "three years imprisonment and a fine up to ₹1 lakh."
        )

    elif question == questions[2]:

        answer = (
            "The section referred to is Section 66C."
        )

    else:

        answer = (
            "Section 66C deals with fraudulent or dishonest "
            "use of another person's password or unique "
            "identification feature."
        )

    print("\nSIMULATED ASSISTANT ANSWER:")
    print(answer)

    conversation.append({
        "role": "user",
        "content": question
    })

    conversation.append({
        "role": "assistant",
        "content": answer
    })
