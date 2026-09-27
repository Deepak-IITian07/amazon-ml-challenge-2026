"""
High-Performance Text Preprocessing and Canonicalization Module
Handles multilingual text normalization, legal suffix identification,
DBA/URL parsing, and address component extraction.
"""

import re
import unidecode
from typing import Set, Tuple, List, Optional


# Precompiled regular expressions for speed
RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_URL_SUFFIX = re.compile(r"\b(com|org|net|in|fr|io|biz|info)\b", re.IGNORECASE)
RE_DBA = re.compile(r"\b(?:doing business as|d/b/a|dba|f/k/a|trading as|t/a)\b", re.IGNORECASE)

# Legal suffixes across US, India, and France
LEGAL_TERMS = {
    # US / UK
    "inc", "incorporated", "llc", "corp", "corporation", "co", "company",
    "ltd", "limited", "lp", "llp", "pllc", "gmbh", "holdings", "holding",
    "enterprises", "enterprise", "group", "services", "solutions",
    # India
    "pvt", "private", "praaivett", "limitted", "limittedd", "elelpi",
    # France
    "sarl", "sas", "sasu", "sci", "eurl", "sa", "snc", "ste", "societe"
}

# Regex to strip legal suffixes at beginning or end of name
_legal_pattern = r"\b(" + "|".join(LEGAL_TERMS) + r")\b"
RE_LEGAL_SUFFIX = re.compile(_legal_pattern, re.IGNORECASE)

# Common address abbreviations
ADDR_ABBR = {
    "st": "street", "saint": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "ct": "court",
    "pl": "place", "pkwy": "parkway", "ln": "lane", "r": "rue", "rue": "rue",
    "ste": "suite", "apt": "apartment", "hwy": "highway", "fl": "floor"
}


def normalize_text(text: Optional[str]) -> str:
    """Basic normalization: transliterate to ASCII, lowercase, strip noise punctuation."""
    if not text or not isinstance(text, str) or text.lower() == "nan" or text.lower() == "null":
        return ""
    # Transliterate unicode to ASCII
    ascii_text = unidecode.unidecode(text).lower()
    # Replace & with and
    ascii_text = ascii_text.replace("&", " and ").replace("@", " at ")
    # Replace hyphens and slashes with spaces to avoid fusing words
    ascii_text = ascii_text.replace("-", " ").replace("/", " ")
    # Strip remaining non-alphanumeric characters
    clean = RE_NON_ALPHANUM.sub(" ", ascii_text)
    # Collapse multiple whitespaces
    return RE_WHITESPACE.sub(" ", clean).strip()


def extract_brand_stem(name: str) -> Tuple[str, Set[str]]:
    """
    Extracts core brand stem and detected legal/structural tokens.
    Handles URLs and DBAs.
    Returns: (cleaned_brand_stem, set_of_extracted_legal_tokens)
    """
    norm = normalize_text(name)
    if not norm:
        return "", set()

    # Check for DBA
    dba_match = RE_DBA.search(norm)
    if dba_match:
        # Take the entity name after DBA if available, else first part
        parts = RE_DBA.split(norm)
        norm = max(parts, key=len).strip()

    # Check for domain names: e.g. "maurewilliamscolombier.com"
    norm = RE_URL_SUFFIX.sub("", norm).strip()

    # Find legal tokens
    tokens = norm.split()
    legal_tokens = set()
    filtered_tokens = []

    for t in tokens:
        if t in LEGAL_TERMS:
            legal_tokens.add(t)
        else:
            filtered_tokens.append(t)

    stem = " ".join(filtered_tokens).strip()
    # If stripping removed everything (e.g. name was literally just "Enterprises LLC"), retain norm
    if not stem:
        stem = norm

    return stem, legal_tokens


def extract_address_components(address: Optional[str]) -> Tuple[str, List[str], Set[str]]:
    """
    Extracts normalized address, list of number anchors (PINs, street nums),
    and normalized address tokens.
    Returns: (normalized_address_str, number_anchors, token_set)
    """
    norm = normalize_text(address)
    if not norm:
        return "", [], set()

    raw_tokens = norm.split()
    # Expand road abbreviations
    expanded_tokens = [ADDR_ABBR.get(t, t) for t in raw_tokens]

    # Extract numbers (building numbers, ZIP, PIN codes)
    numbers = [t for t in expanded_tokens if t.isdigit()]

    clean_addr = " ".join(expanded_tokens)
    return clean_addr, numbers, set(expanded_tokens)


if __name__ == "__main__":
    print("Testing Preprocessing...")
    test_cases = [
        ("Maure Williams Colombier Inc", "85 Wayne Avenue, Ticonderoga, NY"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
        ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu"),
        ("Fluxcira doing business as Grain & Fils", "329 AV. DE DUNKERQUE, LILLE, Hauts-de-France"),
        ("maurewilliamscolombier.com", "Wayne Ave, Ticonderoga Townshiip, New York"),
        ("PAYNE-ENRTPRMISES", "3315 FREMONT SAINT, PEORIA, IL")
    ]

    for name, addr in test_cases:
        stem, legals = extract_brand_stem(name)
        c_addr, nums, tokens = extract_address_components(addr)
        print(f"Original: {name} | {addr}")
        print(f"  Stem: '{stem}', Legals: {legals}")
        print(f"  Addr: '{c_addr}', Nums: {nums}\n")
