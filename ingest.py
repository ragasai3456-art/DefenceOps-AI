import os
import json
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DOCUMENTS_DIR = "documents"
VECTOR_DIR = "vector_db"
DATA_DIR = "data"

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "index.faiss"
)

METADATA_FILE = os.path.join(
    DATA_DIR,
    "metadata.json"
)


# ==========================================
# EMBEDDING MODEL
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text("text").strip()

        if text:

            pages.append({
                "text": text,
                "page": page_number + 1
            })

    doc.close()

    return pages


# ==========================================
# CREATE TEXT CHUNKS
# ==========================================

def create_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# CLASSIFY DOCUMENT
# ==========================================

def classify_document(text):

    text_lower = text.lower()

    # --------------------------------------
    # EQUIPMENT KEYWORDS
    # --------------------------------------

    equipment_keywords = [
        "equipment manual",
        "equipment",
        "operating procedure",
        "operating instructions",
        "starting procedure",
        "start the",
        "does not start",
        "doesn't start",
        "fails to start",
        "troubleshooting",
        "control panel",
        "output switch",
        "selector",
        "generator",
        "component",
        "operating limits",
        "battery module"
    ]

    # --------------------------------------
    # MAINTENANCE KEYWORDS
    # --------------------------------------

    maintenance_keywords = [
        "maintenance",
        "maintenance sop",
        "routine inspection",
        "routine maintenance",
        "cleaning",
        "inspect",
        "inspection",
        "service",
        "maintenance record",
        "maintenance procedure",
        "battery inspection",
        "cleaning procedure",
        "technician review"
    ]

    # --------------------------------------
    # SAFETY KEYWORDS
    # --------------------------------------

    safety_keywords = [
        "safety",
        "hazard",
        "hazards",
        "warning",
        "warnings",
        "precaution",
        "precautions",
        "smoke",
        "leakage",
        "leak",
        "unusual odor",
        "unusual smell",
        "protective",
        "ppe",
        "unsafe condition",
        "electrical safety",
        "ventilation",
        "abnormal heating"
    ]

    # --------------------------------------
    # CALCULATE SCORES
    # --------------------------------------

    equipment_score = sum(
        1
        for keyword in equipment_keywords
        if keyword in text_lower
    )

    maintenance_score = sum(
        1
        for keyword in maintenance_keywords
        if keyword in text_lower
    )

    safety_score = sum(
        1
        for keyword in safety_keywords
        if keyword in text_lower
    )

    scores = {
        "Equipment": equipment_score,
        "Maintenance": maintenance_score,
        "Safety": safety_score
    }

    # --------------------------------------
    # FIND HIGHEST SCORE
    # --------------------------------------

    best_domain = max(
        scores,
        key=scores.get
    )

    # --------------------------------------
    # NO MATCH
    # --------------------------------------

    if scores[best_domain] == 0:

        return "General"

    return best_domain


# ==========================================
# PROCESS ONE PDF
# ==========================================

def process_pdf(pdf_path):

    pages = extract_text_from_pdf(
        pdf_path
    )

    document_name = os.path.basename(
        pdf_path
    )

    # --------------------------------------
    # COMBINE DOCUMENT TEXT
    # --------------------------------------

    full_text = "\n".join(
        page["text"]
        for page in pages
    )

    # --------------------------------------
    # CLASSIFY DOCUMENT
    # --------------------------------------

    domain = classify_document(
        full_text
    )

    print(
        f"Processing: {document_name}"
    )

    print(
        f"Detected Domain: {domain}"
    )

    # --------------------------------------
    # CREATE CHUNKS
    # --------------------------------------

    all_chunks = []

    for page_data in pages:

        chunks = create_chunks(
            page_data["text"]
        )

        for chunk in chunks:

            all_chunks.append({

                "text": chunk,

                "document": document_name,

                "page": page_data["page"],

                "domain": domain

            })

    return all_chunks


# ==========================================
# BUILD VECTOR DATABASE
# ==========================================

def build_vector_database():

    # --------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------

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

    # --------------------------------------
    # FIND PDF FILES
    # --------------------------------------

    pdf_files = [
        file
        for file in os.listdir(
            DOCUMENTS_DIR
        )
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        print(
            "No PDF documents found."
        )

        return 0

    # --------------------------------------
    # PROCESS ALL DOCUMENTS
    # --------------------------------------

    all_chunks = []

    for filename in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        chunks = process_pdf(
            pdf_path
        )

        all_chunks.extend(
            chunks
        )

    # --------------------------------------
    # CHECK CHUNKS
    # --------------------------------------

    if not all_chunks:

        print(
            "No text could be extracted from PDFs."
        )

        return 0

    print(
        f"Total chunks created: {len(all_chunks)}"
    )

    # --------------------------------------
    # EXTRACT TEXT FOR EMBEDDINGS
    # --------------------------------------

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    # --------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------

    print(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # --------------------------------------
    # NORMALIZE EMBEDDINGS
    # --------------------------------------

    faiss.normalize_L2(
        embeddings
    )

    # --------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # --------------------------------------
    # SAVE FAISS INDEX
    # --------------------------------------

    faiss.write_index(
        index,
        INDEX_FILE
    )

    print(
        f"FAISS index saved: {INDEX_FILE}"
    )

    # --------------------------------------
    # SAVE METADATA
    # --------------------------------------

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Metadata saved: {METADATA_FILE}"
    )

    # --------------------------------------
    # FINAL SUMMARY
    # --------------------------------------

    print(
        "\nDocument processing completed."
    )

    print(
        f"Total chunks: {len(all_chunks)}"
    )

    return len(all_chunks)