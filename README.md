# Christian AI Assistant

A scripture-grounded, denomination-aware Christian AI assistant with hallucination prevention, two-pass safety, and Christian-themed image generation. Built on a 100% free tech stack.

## Features

| Capability | Implementation |
|---|---|
| Scripture-grounded answers | RAG over 31,100 KJV verses (ChromaDB + sentence-transformers) |
| Hallucination prevention | Post-generation citation verifier — flags `[UNVERIFIED]` references |
| Denomination awareness | Catholic / Protestant / Baptist / Orthodox / Methodist / Lutheran / Pentecostal |
| Two-pass safety | Keyword rules → LLM classifier (input) + LLM judge (output) |
| Christian image generation | Pollinations.ai (free, no key) with block list + LLM pre-check |
| Conversation memory | SQLite session history (last 20 turns) |
| Adversarial handling | Jailbreaks, verse rewrites, hate content — all blocked before LLM call |

## Tech Stack (all free)

| Component | Tool |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B (free tier) |
| LLM fallback | Gemini 1.5 Flash (free tier) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local) |
| Vector DB | ChromaDB (in-process, local) |
| Bible corpus | Public-domain KJV JSON (31,100 verses) |
| Image generation | Pollinations.ai (no API key required) |
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Memory | SQLite |

## Setup

### 1. Clone and create virtual environment
```bash
cd christian-ai-assistant
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API keys
```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
# GEMINI_API_KEY is optional (used as fallback if Groq quota is hit)
```

### 3. Download Bible corpus and build vector index
```bash
python scripts/download_kjv.py
python scripts/build_index.py
```
This runs once and embeds all 31,100 KJV verses into ChromaDB (~2 min on CPU).

## Running

### Streamlit demo (recommended)
```bash
venv/bin/streamlit run frontend/streamlit_app.py
```
Open **http://localhost:8501**

### FastAPI backend (optional, for API access)
```bash
venv/bin/uvicorn app.main:api --reload --port 8000
```
Swagger docs at **http://localhost:8000/docs**

## Evaluation

Run the 20-case evaluation suite (normal, fake verse, adversarial, contradictory theology, image policy):
```bash
venv/bin/python eval/run_eval.py
```
Current result: **20/20 (100%)**

## Project Structure

```
christian-ai-assistant/
├── app/
│   ├── main.py               # FastAPI app
│   ├── config.py             # API keys, model config, denomination map, block lists
│   ├── routers/
│   │   ├── chat.py           # POST /chat
│   │   └── image.py          # POST /image
│   ├── services/
│   │   ├── pipeline.py       # Orchestrator: routes text vs image, runs all services
│   │   ├── rag.py            # ChromaDB retrieval with denomination filter
│   │   ├── llm.py            # Groq wrapper + system prompt builder
│   │   ├── verifier.py       # Scripture citation verifier (flags hallucinated refs)
│   │   ├── safety.py         # Two-pass input guard + output judge
│   │   ├── image_gen.py      # Pollinations.ai + keyword/LLM safety
│   │   └── memory.py         # SQLite conversation history
│   └── models/
│       └── schemas.py        # Pydantic request/response models
├── data/
│   └── kjv.json              # 31,100 KJV verses (downloaded by setup script)
├── chroma_db/                # Vector index (auto-created by build_index.py)
├── frontend/
│   └── streamlit_app.py      # Streamlit UI
├── eval/
│   ├── eval_dataset.json     # 20 test cases
│   └── run_eval.py           # Automated scorer
├── scripts/
│   ├── download_kjv.py       # Downloads KJV JSON
│   └── build_index.py        # Embeds corpus into ChromaDB
├── requirements.txt
└── .env.example
```

## Key Design Decisions

**1. Verse-level RAG chunks** — each of the 31,100 verses is its own ChromaDB document. This maximises retrieval precision: a query about "love thy neighbour" retrieves exactly the right verse rather than a large block containing unrelated text.

**2. Post-generation scripture verifier** — every `Book Ch:V` pattern in the LLM output is regex-extracted and exact-matched against the KJV corpus before the response is sent. Non-matching references are replaced inline with `[UNVERIFIED: ref]`. This eliminates hallucinated scripture at the source.

**3. Two-stage safety (input + output)** — input-only safety misses subtle drift in the LLM response; output-only safety wastes quota on blocked inputs. Pass 1 (keyword rules, ~0 ms) catches hard violations. Pass 2 (LLM classifier, only on statement-form inputs) catches ambiguous cases. Output judge validates every generated response.

**4. Denomination as system-prompt variable** — injecting denomination into each LLM call avoids hard-coding any single theological tradition. Catholic users get Deuterocanonical books in RAG; Orthodox users get Septuagint framing; Protestant users get Sola Scriptura emphasis.

**5. Pollinations.ai for zero-cost image generation** — no API key, no rate-limit registration, simple HTTP GET. Stem-prefix keyword matching (not exact-word) catches inflected forms ("killing" → "kill", "crusader" → "crusade"). LLM pre-check catches subtle policy violations the block list misses.
