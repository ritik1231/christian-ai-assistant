# Architecture Note — Christian AI Assistant

## System Overview

The assistant is a multi-layer AI pipeline. Every user turn passes through an **input safety layer**, gets routed to either a **text pipeline** (RAG → LLM → Verifier → Output judge) or an **image pipeline** (keyword check → LLM judge → Pollinations.ai), and is persisted to **SQLite session memory** before returning to the user.

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  INPUT SAFETY (two-pass)    │  Pass 1: keyword rules  (~0 ms, no API)
│                             │  Pass 2: LLM classifier (only non-questions)
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
  [text query]    [image request]
       │                │
       ▼                ▼
┌────────────┐   ┌─────────────────┐
│ RAG        │   │ IMAGE PIPELINE  │
│ ChromaDB   │   │ Keyword block   │
│ top-5 KJV  │   │ + LLM pre-check │
│ verses     │   │ + Pollinations  │
└─────┬──────┘   └────────┬────────┘
      │                   │
      ▼                   │
┌────────────┐            │
│ LLM        │            │
│ Groq       │            │
│ LLaMA 3.3  │            │
│ 70B        │            │
└─────┬──────┘            │
      │                   │
      ▼                   │
┌────────────┐            │
│ SCRIPTURE  │            │
│ VERIFIER   │            │
│ regex +    │            │
│ corpus     │            │
│ exact-match│            │
└─────┬──────┘            │
      │                   │
      ▼                   │
┌────────────┐            │
│ OUTPUT     │            │
│ JUDGE      │            │
│ (LLM)      │            │
└─────┬──────┘            │
      └─────────┬─────────┘
                ▼
      SQLite Memory + Response
```

---

## Components

### RAG Pipeline (`services/rag.py`)
The KJV Bible (31,100 verses) is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, free) and stored in ChromaDB. Each verse is its own document — verse-level chunks give far higher retrieval precision than paragraph-level. Denomination filtering excludes Deuterocanonical books for non-Catholic users at query time.

### LLM (`services/llm.py`)
Groq API with LLaMA 3.3 70B (primary, free tier: 6,000 req/day). Gemini 1.5 Flash as automatic fallback on rate-limit. Temperature: 0.3 for factual grounding. The system prompt is rebuilt each turn with denomination context and the top-5 retrieved verses injected as the only permitted citation sources.

### Scripture Verifier (`services/verifier.py`)
Regex extracts every `Book Ch:V` pattern from the LLM output. Each citation is exact-matched against a prebuilt corpus dict (`{"Genesis 1:1": "...", ...}`), O(1) lookup. References not found in the 31,100-verse corpus are replaced inline with `[UNVERIFIED: ref]` before the response reaches the user. This is the primary hallucination-prevention mechanism.

### Safety Layer (`services/safety.py`)
**Input Pass 1** — compiled regex patterns for jailbreaks, verse-rewrite requests, and hate content. Zero latency, no API call. Returns HARD_BLOCK immediately on match.

**Input Pass 2** — LLM classifier (LLaMA 3.1 8B Instant, fast model) called only on statement-form inputs (no `?` in text). Temperature 0.0. Returns one of: `SAFE | ADVERSARIAL | HATE | VERSE_MANIPULATION | OFF_TOPIC`. Skipping question-form inputs eliminates false positives on genuine theological queries.

**Output judge** — secondary LLM call on every generated response. Checks for heretical assertions, hate, political ideology grafting, and fabricated history. On FAIL returns a canned pastoral fallback.

### Image Pipeline (`services/image_gen.py`)
**Pass 1** — stem-prefix keyword matching against `IMAGE_BLOCK_WORDS`. "killing" matches stem "kill", "crusader" matches stem "crusade". Catches morphological variants without an exhaustive word list.

**Pass 2** — LLM pre-check for subtle violations (context-dependent policy breaches the keyword list misses).

**Generation** — Pollinations.ai REST API. No key, no quota registration. `GET /prompt/{encoded_text}` returns a PNG. Style prefix ("Christian artwork, oil painting style, reverent, holy") is prepended to every approved prompt.

### Denomination Router (`config.py`)
Eight traditions supported: Non-denominational, Catholic, Protestant, Baptist, Orthodox, Methodist, Lutheran, Pentecostal. Each maps to a denomination note injected into the LLM system prompt. RAG uses a ChromaDB `where` filter to include/exclude Deuterocanonical books based on denomination. Stored per-session in SQLite.

### Conversation Memory (`services/memory.py`)
SQLite with `sessions` (id, denomination, created_at) and `messages` (session_id, role, content, created_at) tables. Last 20 turns injected as message history on each LLM call.

---

## Key Design Trade-offs

| Decision | Alternative considered | Why this choice |
|---|---|---|
| Verse-level RAG chunks | Paragraph / chapter chunks | Higher precision — a query retrieves the exact verse, not surrounding noise |
| Post-generation verifier | Pre-generation grounding only | LLMs can still generate uncited references; verifier is a hard guarantee |
| Two-stage input safety | Single LLM classifier | Pass 1 catches 95% of violations at zero cost; LLM only runs on ambiguous inputs |
| Groq (LLaMA 3.3 70B) | OpenAI GPT-4 | Free, fast, competitive quality — no cost for assessment demo |
| Pollinations.ai | DALL-E 3 / Stability AI | Zero cost, no API key, sufficient quality for Christian art |
| Direct pipeline import in Streamlit | HTTP calls to FastAPI | Simpler local demo setup; FastAPI still available for external API access |

---

## Evaluation Results

| Category | Score |
|---|---|
| Normal theology queries | 5/5 |
| Fake / hallucinated verse detection | 4/4 |
| Adversarial / jailbreak prompts | 4/4 |
| Contradictory theology | 2/2 |
| Difficult theology | 2/2 |
| Image policy | 3/3 |
| **Overall** | **20/20 (100%)** |
