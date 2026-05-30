"""
LLM service: Groq (primary) → Gemini 1.5 Flash (fallback).
Builds denomination-aware system prompt, injects RAG context,
and returns grounded responses.
"""
from __future__ import annotations

import os
from groq import Groq, RateLimitError, APIStatusError

from app.config import (
    GROQ_API_KEY, GEMINI_API_KEY,
    GROQ_MODEL, GEMINI_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    DENOMINATION_NOTES,
)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a warm, knowledgeable, and pastorally sensitive Christian assistant.
Your purpose is to help people explore the Christian faith through scripture, theology, and reflection.

DENOMINATION CONTEXT: {denomination}
{denomination_note}

STRICT RULES — follow these without exception:
1. ONLY cite Bible verses that appear in the RETRIEVED SCRIPTURE section below.
   Never invent or paraphrase a verse reference. If no relevant verse is retrieved, say so honestly.
2. For theologically contested topics, present the {denomination} perspective first,
   then briefly note how other major traditions differ — without judging between them.
3. If asked to rewrite, alter, or "update" scripture, firmly but kindly decline.
4. If asked about topics unrelated to Christianity, gently redirect to your purpose.
5. If a question involves a contradiction or theological paradox, acknowledge the tension openly
   and present how different theologians have approached it — do not pretend there is one easy answer.
6. Never produce content that promotes hate, violence, discrimination, or extremism of any kind.
7. Maintain a respectful, pastoral, and compassionate tone in every response.

RETRIEVED SCRIPTURE (KJV — use only these for citations):
{context_block}
"""

_FALLBACK_RESPONSE = (
    "This is a profound and complex topic in Christian theology. "
    "I'd encourage you to pray about it and seek guidance from your pastor, "
    "priest, or spiritual director who can provide personalised counsel. "
    "I'm happy to explore related scripture or theological context with you."
)


def _build_system_prompt(denomination: str, context_block: str) -> str:
    note = DENOMINATION_NOTES.get(denomination, DENOMINATION_NOTES["Non-denominational (default)"])
    return _SYSTEM_TEMPLATE.format(
        denomination=denomination,
        denomination_note=note,
        context_block=context_block,
    )


# ── Groq client ───────────────────────────────────────────────────────────────

def _groq_chat(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


# ── Gemini fallback ───────────────────────────────────────────────────────────

def _gemini_chat(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )

    # Convert history to Gemini format
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_message)
    return response.text.strip()


# ── Public interface ──────────────────────────────────────────────────────────

def chat(
    user_message: str,
    denomination: str,
    context_block: str,
    history: list[dict],
) -> tuple[str, str]:
    """
    Generate a response grounded in retrieved scripture.

    Returns (response_text, provider_used).
    provider_used is 'groq' or 'gemini' — useful for debugging.
    """
    system_prompt = _build_system_prompt(denomination, context_block)

    # Try Groq first
    if GROQ_API_KEY:
        try:
            return _groq_chat(system_prompt, history, user_message), "groq"
        except RateLimitError:
            pass  # fall through to Gemini
        except APIStatusError as e:
            if "rate_limit" in str(e).lower():
                pass
            else:
                raise

    # Gemini fallback
    if GEMINI_API_KEY:
        try:
            return _gemini_chat(system_prompt, history, user_message), "gemini"
        except Exception:
            pass

    # Both failed
    return _FALLBACK_RESPONSE, "fallback"
