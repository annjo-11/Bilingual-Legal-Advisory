import streamlit as st

from rag.rag_pipeline import answer_question


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Cyber Law Information Assistant",
    page_icon="⚖️",
    layout="centered"
)


# ==================================================
# SESSION STATE
# ==================================================

# Store the conversation so that follow-up questions
# can refer to previous questions and answers.

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# TITLE
# ==================================================

st.title("⚖️ Cyber Law Information Assistant")

st.write(
    "Ask questions about the Information Technology Act, 2000."
)


# ==================================================
# DISCLAIMER
# ==================================================

st.warning(
    "This system provides informational content based "
    "on the available legal knowledge base. "
    "It is not a substitute for professional legal advice."
)


# ==================================================
# NEW CHAT BUTTON
# ==================================================

if st.button("🗑️ New Chat"):

    st.session_state.messages = []

    st.rerun()


# ==================================================
# DISPLAY PREVIOUS CONVERSATION
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(
            message["content"]
        )

        # Display sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.caption("Sources")

            seen_sources = set()

            for source in message["sources"]:

                document = source.get(
                    "document",
                    "Unknown document"
                )

                page = source.get(
                    "page",
                    "Unknown page"
                )

                source_key = (
                    document,
                    page
                )

                if source_key not in seen_sources:

                    st.write(
                        f"📄 {document} — Page {page}"
                    )

                    seen_sources.add(
                        source_key
                    )


# ==================================================
# USER QUESTION
# ==================================================

question = st.chat_input(
    "Ask your legal question..."
)


# ==================================================
# PROCESS QUESTION
# ==================================================

if question:

    # ----------------------------------------------
    # DISPLAY USER QUESTION
    # ----------------------------------------------

    with st.chat_message("user"):

        st.write(question)


    # ----------------------------------------------
    # BUILD CONVERSATION HISTORY
    # ----------------------------------------------

    conversation_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }

        for message in st.session_state.messages
    ]


    # ----------------------------------------------
    # STORE USER MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ----------------------------------------------
    # GENERATE ANSWER
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the legal knowledge base..."
        ):

            answer, sources = answer_question(
                question,
                conversation_history
            )


        # ------------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------------

        st.write(answer)


        # ------------------------------------------
        # DISPLAY SOURCES
        # ------------------------------------------

        if sources:

            st.caption("Sources")

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

                source_key = (
                    document,
                    page
                )

                if source_key not in seen_sources:

                    st.write(
                        f"📄 {document} — Page {page}"
                    )

                    seen_sources.add(
                        source_key
                    )


    # ----------------------------------------------
    # STORE ASSISTANT RESPONSE
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Knowledge Base: Information Technology Act, 2000"
)