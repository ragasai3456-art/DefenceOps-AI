import os
import re

import streamlit as st

from ingest import build_vector_database
from agent import run_agent


# --------------------------------
# CONFIGURATION
# --------------------------------

DOCUMENTS_DIR = "documents"
VECTOR_DIR = "vector_db"
DATA_DIR = "data"


os.makedirs(
    DOCUMENTS_DIR,
    exist_ok=True
)

os.makedirs(
    VECTOR_DIR,
    exist_ok=True
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="DefenceOps AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("🛡️ DefenceOps AI")

    st.caption(
        "RAG + Agent Intelligence Assistant"
    )

    st.divider()

    st.subheader("System Status")

    st.success("AI Agent Online")

    st.write(
        "Knowledge Base: Ready"
    )

    st.write(
        "Evidence Validation: Enabled"
    )

    st.divider()

    st.subheader("Safety Guardrails")

    st.write(
        "• Answers use uploaded documents only."
    )

    st.write(
        "• Sources and page numbers are verified."
    )

    st.write(
        "• Unsupported questions are refused."
    )

    st.write(
        "• Only synthetic/public documents."
    )


# --------------------------------
# MAIN TITLE
# --------------------------------

st.title(
    "🛡️ DefenceOps AI"
)

st.caption(
    "RAG-Powered Defence Field Operations Intelligence Assistant"
)

st.info(
    "Upload approved documents and ask questions. "
    "The system retrieves relevant evidence and "
    "generates grounded answers."
)


# --------------------------------
# DOCUMENT UPLOAD
# --------------------------------

st.header(
    "📁 Knowledge Base"
)

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        original_name = os.path.basename(
            uploaded_file.name
        )

        safe_name = re.sub(
            r'[<>:"/\\|?*\x00-\x1F]',
            "_",
            original_name
        )

        safe_name = safe_name.strip(
            " ."
        )

        if not safe_name:

            safe_name = (
                "uploaded_document.pdf"
            )

        if not safe_name.lower().endswith(
            ".pdf"
        ):

            safe_name += ".pdf"

        file_path = os.path.abspath(
            os.path.join(
                DOCUMENTS_DIR,
                safe_name
            )
        )

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

    st.success(
        f"{len(uploaded_files)} document(s) uploaded successfully."
    )


# --------------------------------
# PROCESS DOCUMENTS
# --------------------------------

if st.button(
    "⚙️ Process Documents",
    use_container_width=True
):

    with st.spinner(
        "Processing documents..."
    ):

        try:

            count = build_vector_database()

            if count > 0:

                st.success(
                    f"Knowledge base created successfully. "
                    f"{count} chunks indexed."
                )

            else:

                st.warning(
                    "No PDF documents were found."
                )

        except Exception as e:

            st.error(
                f"Document processing failed: {e}"
            )


# --------------------------------
# ASK QUESTION
# --------------------------------

st.divider()

st.header(
    "💬 Ask DefenceOps AI"
)

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What should I check "
        "if the Falcon-1 does not start?"
    ),
    height=100
)


if st.button(
    "🔍 Ask",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Analysing documentation..."
        ):

            try:

                result = run_agent(
                    question.strip()
                )

                # ----------------------------
                # AGENT ANALYSIS
                # ----------------------------

                st.header(
                    "🤖 Agent Analysis"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Detected Domain",
                        result["domain"]
                    )

                with col2:

                    if result["sources"]:

                        st.success(
                            "Evidence Supported"
                        )

                    else:

                        st.warning(
                            "Evidence Insufficient"
                        )

                # ----------------------------
                # ANSWER
                # ----------------------------

                st.header(
                    "📝 Answer"
                )

                if result["sources"]:

                    st.success(
                        result["answer"]
                    )

                else:

                    st.warning(
                        result["answer"]
                    )

                # ----------------------------
                # SOURCES
                # ----------------------------

                if result["sources"]:

                    st.header(
                        "📚 Verified Sources"
                    )

                    for source in result["sources"]:

                        with st.expander(
                            f"📄 {source['document']} — "
                            f"Page {source['page']}"
                        ):

                            st.write(
                                f"**Domain:** "
                                f"{source.get('domain', 'General')}"
                            )

                            st.write(
                                f"**Relevance Score:** "
                                f"{source['score']:.3f}"
                            )

                            st.write(
                                "**Retrieved Evidence:**"
                            )

                            st.write(
                                source["text"]
                            )

            except Exception as e:

                st.error(
                    f"Agent error: {e}"
                )