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
    # If an Ollama API key exists, use Ollama Cloud.
    api_key = get_setting("OLLAMA_API_KEY")

    if api_key:
        model = get_setting("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")

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

    # Otherwise use local Ollama.
    model = get_setting("OLLAMA_MODEL", "llama3.2")

    return ollama.chat(
        model=model,
        messages=messages
    )