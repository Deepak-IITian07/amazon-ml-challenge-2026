"""
Pairwise Feature Extraction Module for Entity Resolution
Extracts lexical, phonetic, structural, and address features between S1 and candidate S2/S3 records.
"""

from typing import List, Set, Tuple, Optional
import rapidfuzz


FEATURE_NAMES = [
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_ratio",
    "name_jaro_winkler",
    "addr_token_set_ratio",
    "addr_ratio",
    "num_common_digits",
    "exact_stem_match",
    "stem_len_diff",
    "is_source3",
    "addr_is_missing",
    "candidate_rank",
    "heuristic_score",
]


def extract_pairwise_features(
    s1_stem: str,
    s1_addr: str,
    s1_nums: Set[str],
    c_stem: str,
    c_addr: str,
    c_nums: Set[str],
    cid: str,
    rank: int,
    heuristic_score: float
) -> List[float]:
    """
    Computes dense numerical feature vector for candidate pair (S1, S2/S3).
    """
    # Name similarities
    n_tsr = rapidfuzz.fuzz.token_set_ratio(s1_stem, c_stem)
    n_sort = rapidfuzz.fuzz.token_sort_ratio(s1_stem, c_stem)
    n_ratio = rapidfuzz.fuzz.ratio(s1_stem, c_stem)
    n_jw = rapidfuzz.distance.JaroWinkler.similarity(s1_stem, c_stem) * 100.0

    # Address similarities
    if c_addr and s1_addr:
        a_tsr = rapidfuzz.fuzz.token_set_ratio(s1_addr, c_addr)
        a_ratio = rapidfuzz.fuzz.ratio(s1_addr, c_addr)
        addr_missing = 0.0
    else:
        a_tsr = 0.0
        a_ratio = 0.0
        addr_missing = 1.0

    # Number anchor agreement
    num_common = float(len(s1_nums & c_nums))

    # Structural features
    exact_stem = 1.0 if s1_stem and s1_stem == c_stem else 0.0
    len_diff = float(abs(len(s1_stem) - len(c_stem)))
    is_s3 = 1.0 if cid.startswith("S3-") else 0.0

    return [
        float(n_tsr),
        float(n_sort),
        float(n_ratio),
        float(n_jw),
        float(a_tsr),
        float(a_ratio),
        num_common,
        exact_stem,
        len_diff,
        is_s3,
        addr_missing,
        float(rank),
        float(heuristic_score)
    ]
