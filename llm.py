import os
import ollama


def get_setting(name, default=None):
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


def chat(messages):
    api_key = get_setting("OLLAMA_API_KEY")

    # Streamlit Cloud / Ollama Cloud
    if api_key:
        model = get_setting(
            "OLLAMA_CLOUD_MODEL",
            "gpt-oss:120b"
        )

        client = ollama.Client(
            host="https://ollama.com",
            headers={
                "Authorization": f"Bearer {api_key}"
            }
        )

        return client.chat(
            model=model,
            messages=messages
        )

    # Local development with Ollama
    model = get_setting(
        "OLLAMA_MODEL",
        "llama3.2"
    )

    return ollama.chat(
        model=model,
        messages=messages
    )