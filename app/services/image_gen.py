"""
Image generation pipeline using Pollinations.ai (free, no API key).

Flow:
  1. Pass 1 — keyword block list (instant, no API call)
  2. Pass 2 — LLM pre-check for subtle policy violations
  3. Prompt sanitiser + style prefix injection
  4. Pollinations.ai HTTP call → returns image URL
"""
from __future__ import annotations

import re
from urllib.parse import quote

import httpx
from groq import Groq

from app.config import (
    GROQ_API_KEY, GROQ_MODEL_FAST,
    POLLINATIONS_BASE, IMAGE_STYLE_PREFIX, IMAGE_BLOCK_WORDS,
)

# ── Canned responses ──────────────────────────────────────────────────────────

BLOCK_MESSAGE = (
    "I'm not able to generate that image — it contains content that doesn't align "
    "with a respectful Christian context. Try describing a scene from scripture, "
    "a nativity, a landscape from the Holy Land, or another peaceful Christian theme."
)

# ── Intent detection ──────────────────────────────────────────────────────────

_IMAGE_TRIGGERS = re.compile(
    r'\b(generate|create|make|draw|show|paint|illustrate|depict|produce|render)'
    r'.{0,20}\b(image|picture|photo|painting|artwork|illustration|drawing|visual)\b',
    re.IGNORECASE,
)

_IMAGE_TRIGGERS_SHORT = re.compile(
    r'\b(image of|picture of|photo of|painting of|artwork of|draw me|show me)\b',
    re.IGNORECASE,
)


def is_image_request(text: str) -> bool:
    """Return True if the user is asking for an image to be generated."""
    return bool(_IMAGE_TRIGGERS.search(text) or _IMAGE_TRIGGERS_SHORT.search(text))


# ── Safety checks ─────────────────────────────────────────────────────────────

def is_image_prompt_safe(prompt: str) -> bool:
    """
    Pass 1 only (stem-prefix keyword check). Used by the eval runner.
    Returns True if safe, False if blocked.

    Uses stem-prefix matching so "killing" catches "kill",
    "crusader" catches "crusade", "violent" catches "violence", etc.
    """
    words = re.findall(r'\b\w+\b', prompt.lower())
    for word in words:
        for blocked in IMAGE_BLOCK_WORDS:
            # Match if the prompt word starts with the blocked stem (min 4 chars)
            stem = blocked[:max(4, len(blocked))]
            if word.startswith(stem):
                return False
    return True


_LLM_IMAGE_JUDGE = """\
You are a content safety reviewer for a Christian image generation service.
Decide if the following image prompt is appropriate to generate.

Reject (reply BLOCK) if the prompt:
- Depicts violence, weapons, blood, or warfare
- Sexualises any figure including religious ones
- Promotes hate, extremism, or discrimination
- Mocks or demeans any religion or group
- Portrays religious figures in a disrespectful or controversial way

Allow (reply ALLOW) if the prompt depicts:
- Scenes from the Bible or Christian history
- Portraits of Jesus, Mary, apostles, saints in reverent contexts
- Christian symbols (cross, dove, fish, etc.)
- Churches, landscapes of the Holy Land, worship scenes
- Abstract representations of faith, hope, love

Image prompt: {prompt}

Reply with ONLY: ALLOW  or  BLOCK:<one sentence reason>"""


def _llm_image_check(prompt: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Catches subtle violations keyword list misses."""
    if not GROQ_API_KEY:
        return True, "no-key"
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL_FAST,
        messages=[{"role": "user", "content": _LLM_IMAGE_JUDGE.format(prompt=prompt)}],
        temperature=0.0,
        max_tokens=40,
    )
    verdict = resp.choices[0].message.content.strip()
    if verdict.upper().startswith("ALLOW"):
        return True, "allow"
    reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
    return False, reason


# ── Prompt sanitiser ──────────────────────────────────────────────────────────

def _sanitise_prompt(prompt: str) -> str:
    """Strip blocked words and inject Christian art style prefix."""
    # Remove any word in the block list
    cleaned = re.sub(
        r'\b(' + '|'.join(re.escape(w) for w in IMAGE_BLOCK_WORDS) + r')\b',
        '',
        prompt,
        flags=re.IGNORECASE,
    ).strip()
    # Collapse extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return IMAGE_STYLE_PREFIX + cleaned


# ── Pollinations.ai client ────────────────────────────────────────────────────

def _call_pollinations(prompt: str, width: int = 768, height: int = 512) -> str:
    """
    Build and validate the Pollinations.ai URL.
    Returns the image URL (the service renders on GET).
    """
    encoded = quote(prompt, safe='')
    url = f"{POLLINATIONS_BASE}/{encoded}?width={width}&height={height}&nologo=true"
    # Do a HEAD request to confirm the endpoint responds before returning the URL
    try:
        with httpx.Client(timeout=5) as client:
            r = client.head(url, follow_redirects=True)
            if r.status_code not in (200, 301, 302, 404):
                raise RuntimeError(f"Pollinations returned {r.status_code}")
    except (httpx.TimeoutException, httpx.ConnectError):
        pass  # return URL anyway; actual fetch happens in the frontend
    return url


# ── Public interface ──────────────────────────────────────────────────────────

def generate(prompt: str) -> dict:
    """
    Full image generation pipeline.

    Returns {
        success:          bool,
        image_url:        str | None,
        sanitised_prompt: str,
        block_reason:     str,
    }
    """
    # Pass 1 — keyword block list
    if not is_image_prompt_safe(prompt):
        return {
            "success": False,
            "image_url": None,
            "sanitised_prompt": prompt,
            "block_reason": "keyword_block",
            "message": BLOCK_MESSAGE,
        }

    # Pass 2 — LLM judge for subtle violations
    llm_safe, llm_reason = _llm_image_check(prompt)
    if not llm_safe:
        return {
            "success": False,
            "image_url": None,
            "sanitised_prompt": prompt,
            "block_reason": f"llm_block: {llm_reason}",
            "message": BLOCK_MESSAGE,
        }

    # Sanitise and generate
    sanitised = _sanitise_prompt(prompt)
    image_url = _call_pollinations(sanitised)

    return {
        "success": True,
        "image_url": image_url,
        "sanitised_prompt": sanitised,
        "block_reason": "",
        "message": "",
    }
