"""
Orchestrates the full pipeline.

Text route:
  1. Input safety (two-pass)
  2. RAG retrieval
  3. LLM generation
  4. Scripture verifier
  5. Output safety judge
  6. Persist to memory

Image route:
  1. Input safety (two-pass)
  2. Image gen (keyword block + LLM judge + Pollinations.ai)
  3. Persist to memory
"""
from app.services.rag import retrieve_verses, format_context_block
from app.services.llm import chat, _FALLBACK_RESPONSE
from app.services.verifier import verify_and_sanitise
from app.services.safety import check_input, check_output, BLOCK_MESSAGES
from app.services.image_gen import is_image_request, generate as generate_image
from app.services.memory import (
    init_db, get_or_create_session, get_denomination,
    get_history, add_message, update_denomination,
)

init_db()


def _base_result(session_id: str, denom: str) -> dict:
    return {
        "reply": "", "sources": [], "unverified": [],
        "session_id": session_id, "denomination": denom,
        "provider": "", "blocked": False, "block_category": "NONE",
        "output_flagged": False, "image_url": None, "is_image": False,
    }


def run(
    user_message: str,
    session_id: str | None = None,
    denomination: str | None = None,
) -> dict:
    """
    Full pipeline for one user turn. Returns a result dict with keys:
      reply, sources, unverified, session_id, denomination,
      provider, blocked, block_category, output_flagged,
      image_url, is_image
    """
    # ── Session / denomination ────────────────────────────────────────────────
    session_id = get_or_create_session(
        session_id, denomination or "Non-denominational (default)"
    )
    if denomination:
        update_denomination(session_id, denomination)
    denom = get_denomination(session_id)
    result = _base_result(session_id, denom)

    # ── Input safety (Pass 1 + Pass 2) ───────────────────────────────────────
    safety = check_input(user_message)
    if not safety.allowed:
        result.update(
            reply=safety.block_message,
            provider="blocked",
            blocked=True,
            block_category=safety.category,
        )
        add_message(session_id, "user", user_message)
        add_message(session_id, "assistant", safety.block_message)
        return result

    # ── Intent routing ────────────────────────────────────────────────────────
    if is_image_request(user_message):
        return _image_route(user_message, session_id, denom, result)
    return _text_route(user_message, session_id, denom, result)


# ── Text route ────────────────────────────────────────────────────────────────

def _text_route(message: str, session_id: str, denom: str, result: dict) -> dict:
    verses = retrieve_verses(message, denomination=denom)
    context_block = format_context_block(verses)
    history = get_history(session_id)

    raw_reply, provider = chat(
        user_message=message,
        denomination=denom,
        context_block=context_block,
        history=history,
    )
    verified_reply, unverified = verify_and_sanitise(raw_reply)

    output_safe, _ = check_output(verified_reply)
    final_reply = verified_reply if output_safe else BLOCK_MESSAGES["OUTPUT_FAIL"]

    add_message(session_id, "user", message)
    add_message(session_id, "assistant", final_reply)

    result.update(
        reply=final_reply,
        sources=verses,
        unverified=unverified,
        provider=provider,
        output_flagged=not output_safe,
    )
    return result


# ── Image route ───────────────────────────────────────────────────────────────

def _image_route(message: str, session_id: str, denom: str, result: dict) -> dict:
    img = generate_image(message)

    if img["success"]:
        reply = (
            f"Here is your Christian-themed image.\n\n"
            f"*Prompt used:* {img['sanitised_prompt']}"
        )
    else:
        reply = img["message"]

    add_message(session_id, "user", message)
    add_message(session_id, "assistant", reply)

    result.update(
        reply=reply,
        is_image=True,
        image_url=img["image_url"],
        provider="pollinations" if img["success"] else "blocked",
        blocked=not img["success"],
        block_category="IMAGE_POLICY" if not img["success"] else "NONE",
    )
    return result
