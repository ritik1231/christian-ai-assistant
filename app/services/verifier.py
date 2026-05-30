"""
Scripture Verifier: extract every Bible citation from LLM output,
look it up in the KJV corpus, and flag any that don't exist.

Handles:
  - Full names:   "Genesis 1:1", "1 Corinthians 13:4"
  - Abbreviations: "Gen 1:1", "1 Cor 13:4", "Ps 23:1"
  - Alternate spellings: "Psalm" / "Psalms", "Song of Solomon" / "Song of Songs"
  - Fake books:   "Hezekiah 4:8"  → [UNVERIFIED]
  - Hallucinated chapter/verse numbers beyond canon bounds → [UNVERIFIED]
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from app.config import KJV_PATH

# ── Canonical book name aliases ───────────────────────────────────────────────
# Maps every common abbreviation / alternate name → canonical name in corpus
BOOK_ALIASES: dict[str, str] = {
    # Pentateuch
    "gen": "Genesis", "gn": "Genesis",
    "exo": "Exodus", "ex": "Exodus",
    "lev": "Leviticus", "lv": "Leviticus",
    "num": "Numbers", "nm": "Numbers",
    "deu": "Deuteronomy", "deut": "Deuteronomy", "dt": "Deuteronomy",
    # History
    "jos": "Joshua", "josh": "Joshua",
    "jdg": "Judges", "judg": "Judges",
    "rut": "Ruth",
    "1sa": "1 Samuel", "1sam": "1 Samuel", "1 sam": "1 Samuel",
    "2sa": "2 Samuel", "2sam": "2 Samuel", "2 sam": "2 Samuel",
    "1ki": "1 Kings", "1kgs": "1 Kings", "1 kings": "1 Kings",
    "2ki": "2 Kings", "2kgs": "2 Kings", "2 kings": "2 Kings",
    "1ch": "1 Chronicles", "1chr": "1 Chronicles", "1 chr": "1 Chronicles",
    "2ch": "2 Chronicles", "2chr": "2 Chronicles", "2 chr": "2 Chronicles",
    "ezr": "Ezra",
    "neh": "Nehemiah",
    "est": "Esther",
    # Poetry
    "job": "Job",
    "ps": "Psalms", "psa": "Psalms", "psalm": "Psalms", "pss": "Psalms",
    "pro": "Proverbs", "prov": "Proverbs", "prv": "Proverbs",
    "ecc": "Ecclesiastes", "eccl": "Ecclesiastes", "qoh": "Ecclesiastes",
    "sos": "Song of Solomon", "song": "Song of Solomon",
    "song of songs": "Song of Solomon", "sg": "Song of Solomon",
    # Major Prophets
    "isa": "Isaiah", "is": "Isaiah",
    "jer": "Jeremiah",
    "lam": "Lamentations",
    "eze": "Ezekiel", "ezek": "Ezekiel",
    "dan": "Daniel",
    # Minor Prophets
    "hos": "Hosea",
    "joe": "Joel", "jl": "Joel",
    "amo": "Amos",
    "oba": "Obadiah",
    "jon": "Jonah",
    "mic": "Micah",
    "nah": "Nahum",
    "hab": "Habakkuk",
    "zep": "Zephaniah", "zeph": "Zephaniah",
    "hag": "Haggai",
    "zec": "Zechariah", "zech": "Zechariah",
    "mal": "Malachi",
    # Gospels / Acts
    "mat": "Matthew", "matt": "Matthew", "mt": "Matthew",
    "mar": "Mark", "mrk": "Mark", "mk": "Mark",
    "luk": "Luke", "lk": "Luke",
    "joh": "John", "jn": "John",
    "act": "Acts",
    # Epistles
    "rom": "Romans",
    "1co": "1 Corinthians", "1cor": "1 Corinthians", "1 cor": "1 Corinthians",
    "2co": "2 Corinthians", "2cor": "2 Corinthians", "2 cor": "2 Corinthians",
    "gal": "Galatians",
    "eph": "Ephesians",
    "phi": "Philippians", "php": "Philippians", "phil": "Philippians",
    "col": "Colossians",
    "1th": "1 Thessalonians", "1thes": "1 Thessalonians",
    "2th": "2 Thessalonians", "2thes": "2 Thessalonians",
    "1ti": "1 Timothy", "1tim": "1 Timothy",
    "2ti": "2 Timothy", "2tim": "2 Timothy",
    "tit": "Titus",
    "phm": "Philemon",
    "heb": "Hebrews",
    "jam": "James", "jas": "James",
    "1pe": "1 Peter", "1pet": "1 Peter",
    "2pe": "2 Peter", "2pet": "2 Peter",
    "1jo": "1 John", "1jn": "1 John",
    "2jo": "2 John", "2jn": "2 John",
    "3jo": "3 John", "3jn": "3 John",
    "jud": "Jude",
    "rev": "Revelation", "re": "Revelation",
}

# Regex: optional leading digit, word(s) for book name, space, ch:v
# Matches: "John 3:16" / "1 Corinthians 13:4-7" / "Gen. 1:1" / "Ps 23:1"
_CITATION_RE = re.compile(
    r'\b(\d\s?)?([A-Z][a-zA-Z]+(?:\s+of\s+[A-Za-z]+)?\.?)'
    r'\s+(\d{1,3}):(\d{1,3})(?:-\d{1,3})?',
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _build_corpus() -> dict[str, str]:
    """Returns {ref_string: verse_text}, e.g. {"Genesis 1:1": "In the beginning..."}"""
    with open(KJV_PATH, encoding="utf-8") as f:
        kjv = json.load(f)

    corpus: dict[str, str] = {}
    for book in kjv:
        name = book["name"]
        for ch_idx, chapter in enumerate(book["chapters"], start=1):
            for v_idx, text in enumerate(chapter, start=1):
                key = f"{name} {ch_idx}:{v_idx}"
                corpus[key] = text.strip()
    return corpus


def _normalise_book(raw: str) -> str:
    """Resolve a raw book string (possibly abbreviated) to a canonical name."""
    clean = raw.strip().rstrip(".").lower()
    if clean in BOOK_ALIASES:
        return BOOK_ALIASES[clean]
    # Try title-case lookup (handles "Genesis", "Revelation", etc. directly)
    titled = raw.strip().rstrip(".")
    return titled  # may not be canonical — corpus lookup will catch it


def verify_and_sanitise(text: str) -> tuple[str, list[str]]:
    """
    Scan `text` for Bible citations and verify each against the KJV corpus.
    Returns (sanitised_text, list_of_unverified_refs).

    Unverified citations are replaced inline with [UNVERIFIED: <ref>].
    """
    corpus = _build_corpus()
    unverified: list[str] = []

    def _replace(match: re.Match) -> str:
        prefix_digit = (match.group(1) or "").strip()
        book_raw = match.group(2).strip()
        ch = match.group(3)
        v = match.group(4)

        # Reconstruct the book name
        full_book_raw = f"{prefix_digit} {book_raw}".strip() if prefix_digit else book_raw
        canonical_book = _normalise_book(full_book_raw)
        ref = f"{canonical_book} {ch}:{v}"

        if ref in corpus:
            return match.group(0)  # verified — leave unchanged

        unverified.append(ref)
        return f"[UNVERIFIED: {ref}]"

    sanitised = _CITATION_RE.sub(_replace, text)
    return sanitised, unverified


def lookup_verse(ref: str) -> str | None:
    """Direct verse lookup. Returns verse text or None if not found."""
    corpus = _build_corpus()
    return corpus.get(ref)
