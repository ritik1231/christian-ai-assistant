import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Disable ChromaDB anonymous telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

BASE_DIR = Path(__file__).resolve().parent.parent

# ── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Model config ──────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-1.5-flash"

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
HISTORY_TURNS = 20          # turns kept in conversation memory
RAG_TOP_K = 5               # verses retrieved per query

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
KJV_PATH = DATA_DIR / "kjv.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Denomination config ───────────────────────────────────────────────────────
DENOMINATIONS = [
    "Non-denominational (default)",
    "Catholic",
    "Protestant",
    "Baptist",
    "Orthodox",
    "Methodist",
    "Lutheran",
    "Pentecostal",
]

# Deuterocanonical books included only for Catholic tradition
DEUTEROCANONICAL_BOOKS = {
    "Tobit", "Judith", "1 Maccabees", "2 Maccabees",
    "Wisdom", "Sirach", "Baruch",
}

DENOMINATION_NOTES = {
    "Catholic": (
        "Include Deuterocanonical books. Affirm Purgatory, Marian doctrines (Immaculate Conception, "
        "Assumption), and Papal authority. Cite the Catechism of the Catholic Church where relevant."
    ),
    "Protestant": (
        "Use the 66-book Protestant canon only. Frame answers around Sola Scriptura, Sola Fide, "
        "and salvation by grace through faith alone."
    ),
    "Baptist": (
        "Protestant base. Emphasise believer's baptism by immersion, local church autonomy, "
        "and the priesthood of all believers. Do not affirm infant baptism."
    ),
    "Orthodox": (
        "Include Septuagint books. Affirm Theosis, Holy Tradition alongside Scripture, "
        "the seven Ecumenical Councils, and the distinct Orthodox position on the Filioque."
    ),
    "Methodist": (
        "Protestant base with Wesleyan emphasis on prevenient grace, sanctification, "
        "and the possibility of entire sanctification."
    ),
    "Lutheran": (
        "Protestant base. Affirm consubstantiation (Real Presence) in the Eucharist, "
        "Law and Gospel distinction, and Luther's two kingdoms doctrine."
    ),
    "Pentecostal": (
        "Protestant base with emphasis on baptism of the Holy Spirit, speaking in tongues "
        "as initial evidence, divine healing, and the continuation of spiritual gifts."
    ),
    "Non-denominational (default)": (
        "Use a neutral, ecumenical framing. Present the major Christian perspectives on "
        "contested topics without adjudicating between them."
    ),
}

# ── Image generation ──────────────────────────────────────────────────────────
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
IMAGE_STYLE_PREFIX = "Christian artwork, oil painting style, reverent, holy, "

IMAGE_BLOCK_WORDS = {
    "weapon", "weapons", "gun", "guns", "sword", "swords", "blood", "gore",
    "war", "warfare", "crusade", "crusades", "violence", "violent",
    "naked", "nude", "nudity", "sexual", "sexy",
    "demon", "demonic", "satanic", "satan", "devil", "occult",
    "cult", "heretic", "heresy", "torture", "kill", "murder",
    "hate", "racist", "extremist",
}
