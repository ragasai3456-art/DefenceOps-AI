import os
import time

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

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]

    last_error = None

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            return {
                "message": {
                    "content": response.text
                }
            }

        except Exception as error:
            last_error = error

            # Try the next model if this model is temporarily unavailable
            if "503" in str(error) or "UNAVAILABLE" in str(error):
                time.sleep(1)
                continue

            raise error

    raise RuntimeError(
        f"All Gemini models are temporarily unavailable. "
        f"Last error: {last_error}"
    )