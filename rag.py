import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


INDEX_FILE = "vector_db/index.faiss"
METADATA_FILE = "data/metadata.json"


model = SentenceTransformer("all-MiniLM-L6-v2")


def search_documents(question, domain="General", top_k=5):

    # ==========================================
    # CHECK DATABASE
    # ==========================================

    if not os.path.exists(INDEX_FILE):
        return []

    if not os.path.exists(METADATA_FILE):
        return []

    # ==========================================
    # LOAD DATABASE
    # ==========================================

    index = faiss.read_index(INDEX_FILE)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if not metadata:
        return []

    # ==========================================
    # CREATE QUESTION EMBEDDING
    # ==========================================

    question_embedding = model.encode(
        [question],
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(question_embedding)

    # ==========================================
    # SEARCH ALL CHUNKS
    # ==========================================

    search_count = len(metadata)

    scores, indices = index.search(
        question_embedding,
        search_count
    )

    results = []

    # ==========================================
    # DOMAIN FILTER
    # ==========================================

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        result = metadata[index_id].copy()

        result_domain = result.get(
            "domain",
            "General"
        )

        # Search only the routed domain
        if domain != "General":

            if result_domain != domain:
                continue

        # ======================================
        # KEEP CHUNKS
        # ======================================

        result["score"] = float(score)

        results.append(result)

        # We intentionally DO NOT remove chunks
        # from the same document/page.
        #
        # One page can contain multiple chunks,
        # and different chunks may contain different
        # information relevant to the question.

        if len(results) >= top_k:
            break

    return results


def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
EVIDENCE CHUNK {i}

SOURCE DOCUMENT:
{result['document']}

PAGE:
{result['page']}

DOCUMENT DOMAIN:
{result.get('domain', 'General')}

RELEVANCE SCORE:
{result['score']:.3f}

CONTENT:
{result['text']}
"""
        )

    return "\n-------------------------\n".join(
        context_parts
    )