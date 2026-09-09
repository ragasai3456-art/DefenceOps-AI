import os

from google import genai


def get_setting(name, default=None):
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


def chat(messages):
    api_key = get_setting("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = genai.Client(api_key=api_key)

    # Convert the existing message format
    # used by our agent into one prompt.
    prompt_parts = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            prompt_parts.append(
                f"SYSTEM INSTRUCTION:\n{content}"
            )
        else:
            prompt_parts.append(
                f"USER:\n{content}"
            )

    prompt = "\n\n".join(prompt_parts)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return {
        "message": {
            "content": response.text
        }
    }