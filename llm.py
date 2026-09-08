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


def is_streamlit_cloud():
    try:
        import streamlit as st

        return "LLM_PROVIDER" in st.secrets
    except Exception:
        return False


def chat(messages):

    if is_streamlit_cloud():
        provider = "cloud"
    else:
        provider = get_setting("LLM_PROVIDER", "local")

    if provider.lower() == "local":

        model = get_setting(
            "OLLAMA_MODEL",
            "llama3.2"
        )

        return ollama.chat(
            model=model,
            messages=messages
        )

    if provider.lower() == "cloud":

        api_key = get_setting("OLLAMA_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY is not configured."
            )

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

    raise RuntimeError(
        f"Unknown LLM_PROVIDER: {provider}"
    )