"""
Text normalization for business_name and business_address.

Goal: turn noisy, inconsistent strings into a canonical form so that
"Corp" / "Corporation", "Rd" / "Road", "&" / "and" etc. stop looking like
different businesses. This is pure string processing - no external lookups,
no APIs, nothing that touches the network (required by the challenge rules).

Everything here is deterministic and offline.
"""

import re
import unicodedata

# Tokens that are legal-entity noise, not part of the business's identity.
# Stripped out because they vary a lot between sources ("Pvt Ltd" vs
# "Private Limited" vs nothing at all) without changing which business it is.
LEGAL_SUFFIX_TOKENS = {
    "pvt", "ltd", "llp", "llc", "inc", "incorporated", "corporation",
    "corp", "co", "company", "limited", "private", "enterprises",
    "enterprise", "industries", "industry", "group", "solutions",
    "services", "sarl", "gmbh", "sa", "plc",
}

# Common English stopwords that don't help disambiguate one business from
# another.
STOPWORDS = {"the", "and", "of", "a", "an", "&"}

# Address abbreviation normalization: applied as whole-word regex swaps.
ADDRESS_ABBREV = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bcourt\b": "ct",
    r"\bsquare\b": "sq",
    r"\bnear\b": "",
    r"\bopp\.?\b": "",
    r"\bopposite\b": "",
    r"\bno\.\b": "no",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics from Latin-script text (café -> cafe).
    Leaves non-Latin scripts (e.g. Devanagari) unchanged - transliterating
    those reliably needs a proper transliteration table, which is a good
    future improvement but out of scope for a first working pipeline.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _basic_clean(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _strip_accents(text)
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name) -> str:
    """Canonical form of a business name: lowercased, accent-stripped,
    punctuation removed, legal-suffix / stopword tokens dropped."""
    text = _basic_clean(name)
    if not text:
        return ""
    tokens = [
        t for t in text.split()
        if t not in STOPWORDS and t not in LEGAL_SUFFIX_TOKENS
    ]
    return " ".join(tokens)


def normalize_address(address) -> str:
    """Canonical form of an address: lowercased, accent-stripped,
    common abbreviations applied, landmark filler words removed."""
    text = _basic_clean(address)
    if not text:
        return ""
    for pattern, repl in ADDRESS_ABBREV.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set:
    return set(text.split()) if text else set()