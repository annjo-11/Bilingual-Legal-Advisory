import streamlit as st

from rag.rag_pipeline import answer_question


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cyber Law Information Assistant",
    page_icon="⚖️",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚖️ Cyber Law Information Assistant")

st.write(
    "Ask questions about the Information Technology Act, 2000."
)


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.warning(
    "This system provides informational content based "
    "on the available legal knowledge base. "
    "It is not a substitute for professional legal advice."
)


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is identity theft under Section 66C?"
)


# --------------------------------------------------
# ASK BUTTON
# --------------------------------------------------

if st.button("Ask Question"):

    if question.strip() == "":
        
        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the legal knowledge base..."
        ):

            answer, sources = answer_question(
                question
            )


        # ------------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------------

        st.subheader("Answer")

        st.write(answer)


        # ------------------------------------------
        # DISPLAY SOURCES
        # ------------------------------------------

        st.subheader("Sources")

        seen_sources = set()

        for source in sources:
            document = source.get(
                "document",
                "Unknown document"
            )

            page = source.get(
                 "page",
                 "Unknown page"
                 )

            source_key = (document, page)

            if source_key not in seen_sources:
                st.write(
                    f"📄 {document} — Page {page}"
                    )

                seen_sources.add(source_key)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Knowledge Base: Information Technology Act, 2000"
)