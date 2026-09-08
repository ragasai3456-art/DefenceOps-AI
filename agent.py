from llm import chat
from router import route_question
from rag import search_documents, build_context


INSUFFICIENT_MESSAGE = (
    "The available documentation is insufficient to answer this question."
)


def run_agent(question):

    print("\n================ AGENT DEBUG ================")

    # 1. ROUTER
    domain = route_question(question)

    print("DOMAIN:", domain)

    # 2. RETRIEVAL
    results = search_documents(
        question,
        domain=domain,
        top_k=3
    )

    print("RESULT COUNT:", len(results))

    if results:
        print("TOP DOCUMENT:", results[0].get("document"))
        print("TOP PAGE:", results[0].get("page"))
        print("TOP DOMAIN:", results[0].get("domain"))
        print("TOP SCORE:", results[0].get("score"))

    print("=============================================\n")

    # 3. NO RESULTS
    if not results:

        return {
            "domain": domain,
            "answer": INSUFFICIENT_MESSAGE,
            "sources": []
        }

    # 4. BUILD CONTEXT
    context = build_context(results)

    # 5. GENERATE ANSWER
    prompt = f"""
You are DefenceOps AI.

Answer the user's question using ONLY the
documentation provided below.

QUESTION:
{question}

DOCUMENTATION:
{context}

RULES:

- Use only the documentation.
- Do not use outside knowledge.
- Do not guess.
- Do not invent information.
- Do not invent procedures.
- Do not invent specifications.
- Ignore unrelated information.
- Answer the exact question.
- Keep the answer concise.

If the documentation genuinely does not contain
the answer, say:

The available documentation is insufficient to answer this question.

ANSWER:
"""

    response = chat(
        [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"].strip()

    if not answer:
        answer = INSUFFICIENT_MESSAGE

    # IMPORTANT:
    # Keep retrieved sources.
    return {
        "domain": domain,
        "answer": answer,
        "sources": results
    }