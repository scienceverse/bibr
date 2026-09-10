"""Per-field comparison metrics for evaluating bibr extraction against ground truth.

All metric functions return a float in [0, 1] where higher is better.
No external dependencies beyond rapidfuzz (already a core bibr dep).

Evaluation methodology follows GROBID (Levenshtein matching, cascade reference
matching) and OmniDocBench (Normalized Edit Distance) best practices.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio

# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------


def _normalize_unicode(text: str) -> str:
    """NFKD normalize and strip combining characters (diacritics)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _strip_punctuation(text: str) -> str:
    """Remove punctuation and collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# DOI normalization
# ---------------------------------------------------------------------------

_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)

# URL-like patterns that get concatenated onto DOIs by OCR
_DOI_EMBEDDED_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)


def normalize_doi(doi: str) -> str:
    """Normalize a DOI by stripping URL prefixes, ``doi:`` prefix, embedded URL
    suffixes, URL-encoded characters, and lowercasing."""
    if not doi:
        return ""
    doi = doi.strip().lower()
    if doi.startswith("doi:"):
        doi = doi[4:].lstrip()
    for prefix in _DOI_URL_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    # Handle URL-encoded characters
    doi = doi.replace("%2f", "/")
    # Strip embedded URL suffixes (OCR concatenation artifacts)
    slash_pos = doi.find("/")
    if slash_pos >= 0:
        suffix = doi[slash_pos + 1 :]
        m = _DOI_EMBEDDED_URL_RE.search(suffix)
        if m:
            doi = doi[: slash_pos + 1 + m.start()]
    # Strip trailing punctuation and slashes
    doi = doi.rstrip(".,;: /")
    # Collapse repeated slashes (done after URL stripping so "https://" isn't normalized first)
    doi = re.sub(r"/+", "/", doi)
    return doi


# ---------------------------------------------------------------------------
# Globally-greedy bipartite matching
# ---------------------------------------------------------------------------


def _greedy_match_count(pairs: list[tuple[float, int, int]]) -> int:
    """Globally-greedy matching: pick highest-scoring pair, mark both sides used, repeat.

    Args:
        pairs: list of (score, gt_index, ext_index) tuples.

    Returns:
        Number of matched pairs.
    """
    pairs.sort(key=lambda x: -x[0])
    matched_gt: set[int] = set()
    matched_ext: set[int] = set()
    count = 0
    for _score, gi, ei in pairs:
        if gi not in matched_gt and ei not in matched_ext:
            matched_gt.add(gi)
            matched_ext.add(ei)
            count += 1
    return count


def _greedy_match_pairs(pairs: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    """Globally-greedy matching returning the matched (gt_idx, ext_idx) pairs."""
    pairs.sort(key=lambda x: -x[0])
    matched_gt: set[int] = set()
    matched_ext: set[int] = set()
    result = []
    for _score, gi, ei in pairs:
        if gi not in matched_gt and ei not in matched_ext:
            matched_gt.add(gi)
            matched_ext.add(ei)
            result.append((gi, ei))
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Length of the longest common subsequence between two token lists."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ---------------------------------------------------------------------------
# Title metrics
# ---------------------------------------------------------------------------


def title_exact_match(extracted: str, ground_truth: str) -> float | None:
    """1.0 if titles match (case-insensitive, stripped), else 0.0.

    None when the ground truth has no title — nothing to evaluate against.
    """
    if not ground_truth.strip():
        return None
    return 1.0 if extracted.strip().lower() == ground_truth.strip().lower() else 0.0


def title_similarity(extracted: str, ground_truth: str) -> float | None:
    """Normalized Levenshtein similarity between titles (0-1).

    Uses Normalized Levenshtein Distance (standard in GROBID and OmniDocBench)
    rather than Jaro-Winkler (designed for short person names, not titles).
    None when the ground truth has no title.
    """
    if not ground_truth.strip():
        return None
    return Levenshtein.normalized_similarity(extracted.strip(), ground_truth.strip())


# A containment match must retain at least this fraction of the longer title.
# Without it the rule would forgive a dropped translation or subtitle of any
# length, even when the remaining title is less than half of the original.
_MIN_CONTAINMENT_LENGTH_RATIO = 0.7


def title_soft_match(extracted: str, ground_truth: str) -> float | None:
    """1.0 if titles match after stripping punctuation and normalizing whitespace.

    Follows GROBID's "soft" matching mode: ignores punctuation, extra whitespace,
    and case differences.  Also returns 1.0 when the shorter title is a
    word-aligned prefix of the longer one *and* retains at least
    ``_MIN_CONTAINMENT_LENGTH_RATIO`` of it — a tolerance for a trailing-word
    difference, not for a dropped subtitle. The rule was originally written for
    Crossref's SAGE-style title/subtitle split (GT truncated at the colon); gold
    is now PDF-verbatim and Crossref-depolluted, so with the sides symmetric the
    only thing it can still forgive is bibr truncating its own output. The
    shorter side must also be substantial enough not to match coincidentally
    (see ``_containment_eligible``).

    None when the ground truth has no title.
    """
    if not ground_truth.strip():
        return None
    e = _strip_punctuation(extracted.lower())
    g = _strip_punctuation(ground_truth.lower())
    if e == g:
        return 1.0
    return 1.0 if _is_containment_match(e, g) else 0.0


def title_soft_containment(extracted: str, ground_truth: str) -> float | None:
    """1.0 when ``title_soft_match`` scored 1.0 via containment, not exact match.

    Reported per paper so the aggregate makes reliance on the containment
    branch visible: summed over a run it is ``title_soft_containment_n``, the
    number of papers whose title_soft point rests on the tolerance rather than
    on a real match. None when the ground truth has no title (same exclusion as
    ``title_soft_match``).
    """
    if not ground_truth.strip():
        return None
    e = _strip_punctuation(extracted.lower())
    g = _strip_punctuation(ground_truth.lower())
    if e == g:
        return 0.0
    return 1.0 if _is_containment_match(e, g) else 0.0


def _is_containment_match(e: str, g: str) -> bool:
    """Shorter title is a word-aligned prefix of the longer, and long enough."""
    short, long = (g, e) if len(g) <= len(e) else (e, g)
    if not _containment_eligible(short):
        return False
    if len(short) < _MIN_CONTAINMENT_LENGTH_RATIO * len(long):
        return False
    return long.startswith(short) and (len(long) == len(short) or long[len(short)] == " ")


# Scripts written without inter-word spaces (Han, hiragana/katakana, Thai).
# str.split() returns a single token for these however long the string is, so a
# word-count gate can never fire for them.
_SCRIPTIO_CONTINUA_RE = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿฀-๿]")

# Character floor standing in for "at least two words" where words are not
# space-delimited. Two Latin words run ~10 characters; 6 CJK characters is
# already a substantive phrase, well clear of coincidental-prefix territory.
_MIN_CONTINUA_PREFIX_CHARS = 6


def _containment_eligible(short: str) -> bool:
    """Is ``short`` substantial enough that a prefix match means something?

    Guards the containment rule against false positives on very short strings.
    The original two-word test silently excluded scriptio-continua scripts: a
    Japanese main-title/subtitle split scored 0.0 while the byte-for-byte
    equivalent Latin split scored 1.0, because a spaceless Japanese title is one
    ``split()`` token. Those scripts get an equivalent character floor instead.
    """
    if len(short.split()) >= 2:
        return True
    return bool(_SCRIPTIO_CONTINUA_RE.search(short)) and len(short) >= _MIN_CONTINUA_PREFIX_CHARS


# ---------------------------------------------------------------------------
# DOI metric
# ---------------------------------------------------------------------------


def doi_match(extracted: str, ground_truth: str) -> float | None:
    """1.0 if DOIs match after normalization, else 0.0.

    Normalizes URL prefixes, doi: prefix, URL-encoding, embedded URL suffixes,
    trailing punctuation, and case before comparison.

    None only when *neither* side has a DOI: the paper prints none and the
    extraction asserts none, so there is nothing to compare and the paper is
    excluded rather than penalized. Asserting a DOI against a page that prints
    none is a wrong answer, not an abstention, and scores 0.0 — otherwise the
    metric is recall-only and an invented DOI costs nothing.
    """
    gt = normalize_doi(ground_truth)
    ext = normalize_doi(extracted)
    if not gt:
        return None if not ext else 0.0
    return 1.0 if ext == gt else 0.0


# ---------------------------------------------------------------------------
# Abstract metrics
# ---------------------------------------------------------------------------


def abstract_rouge_l(extracted: str, ground_truth: str) -> float | None:
    """ROUGE-L F1 score between abstracts (0-1). Uses whitespace tokenization + LCS.

    None when the ground truth has no abstract.
    """
    ext_tokens = extracted.lower().split()
    gt_tokens = ground_truth.lower().split()

    if not gt_tokens:
        return None
    if not ext_tokens:
        return 0.0

    lcs_len = _lcs_length(ext_tokens, gt_tokens)

    precision = lcs_len / len(ext_tokens)
    recall = lcs_len / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def abstract_ned(extracted: str, ground_truth: str) -> float | None:
    """Normalized Edit Distance similarity for abstracts (0-1).

    Standard metric used by OmniDocBench. Computed as 1 - NED where
    NED = levenshtein_distance / max(len_a, len_b).

    None when the ground truth has no abstract.
    """
    if not ground_truth.strip():
        return None
    if not extracted.strip():
        return 0.0
    return Levenshtein.normalized_similarity(
        extracted.strip().lower(), ground_truth.strip().lower()
    )


# ---------------------------------------------------------------------------
# Keyword metrics
# ---------------------------------------------------------------------------


def keywords_f1(extracted: list[str], ground_truth: list[str]) -> float:
    """F1 score on lowercased keyword sets (0-1)."""
    ext_set = {k.lower().strip() for k in extracted if k}
    gt_set = {k.lower().strip() for k in ground_truth if k}

    if not ext_set and not gt_set:
        return 1.0
    if not ext_set or not gt_set:
        return 0.0

    intersection = len(ext_set & gt_set)
    precision = intersection / len(ext_set)
    recall = intersection / len(gt_set)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def _normalize_keyword(kw: str) -> str:
    """Normalize a keyword for fuzzy comparison: lowercase, strip, replace hyphens."""
    return kw.lower().strip().replace("-", " ").replace("\u2013", " ").replace("\u2014", " ")


def keywords_fuzzy_f1(
    extracted: list[str], ground_truth: list[str], threshold: float = 80.0
) -> float:
    """F1 on keywords with fuzzy matching (token_sort_ratio >= threshold).

    Uses globally-greedy matching on the similarity matrix instead of
    sequential greedy to avoid order-dependent matching artifacts.
    """
    ext_list = [_normalize_keyword(k) for k in extracted if k]
    gt_list = [_normalize_keyword(k) for k in ground_truth if k]

    if not ext_list and not gt_list:
        return 1.0
    if not ext_list or not gt_list:
        return 0.0

    # Build pairwise scores above threshold and match globally
    pairs = []
    for gi, gt_kw in enumerate(gt_list):
        for ei, ext_kw in enumerate(ext_list):
            score = token_sort_ratio(gt_kw, ext_kw)
            if score >= threshold:
                pairs.append((score, gi, ei))

    tp = _greedy_match_count(pairs)

    precision = tp / len(ext_list)
    recall = tp / len(gt_list)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Author metrics
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Normalize a name: lowercase, strip diacritics, strip whitespace."""
    return _normalize_unicode(name.lower().strip())


def _name_similarity(ext_name: str, gt_name: str) -> float:
    """Order-insensitive Jaro-Winkler similarity between two normalized names.

    Compared both as printed and with the tokens of each side sorted, taking the
    better of the two. Raw Jaro-Winkler is order-sensitive: a family-first
    byline ("Oloda Oluwatayo Felix" vs "Oluwatayo Felix Oloda") scores 0.837,
    below the match threshold, purely because of token order. Sorting alone
    would fix that but costs the alignment tolerance the raw form gives to an
    inserted or dropped middle name, so both orderings are tried.
    """
    sim = JaroWinkler.normalized_similarity(ext_name, gt_name)
    ext_sorted = " ".join(sorted(ext_name.split()))
    gt_sorted = " ".join(sorted(gt_name.split()))
    if ext_sorted == ext_name and gt_sorted == gt_name:
        return sim
    return max(sim, JaroWinkler.normalized_similarity(ext_sorted, gt_sorted))


def _token_covered(gt_token: str, ext_tokens: list[str], threshold: float) -> bool:
    """Is one gold name token accounted for by some extracted token?

    Initial-vs-full-given ("j" for "john") counts as covered — either side may
    be the abbreviation — but an unrelated token does not.
    """
    for et in ext_tokens:
        if JaroWinkler.normalized_similarity(gt_token, et) >= threshold:
            return True
        if len(et) < len(gt_token) and gt_token.startswith(et):
            return True
        if len(gt_token) < len(et) and et.startswith(gt_token):
            return True
    return False


def _tokens_covered(ext_name: str, gt_name: str, threshold: float) -> bool:
    """Every gold token of a shorter candidate's counterpart must be accounted for.

    Jaro-Winkler's prefix boost scores a *completely dropped surname* at
    0.92-0.947 ("kaisin" vs "kaisin yee"), so a candidate with fewer name tokens
    than gold must not be allowed to match on prefix alone. Single-character
    gold tokens (middle initials) are exempt — dropping one is a formatting
    difference, not a lost name.
    """
    # Punctuation-stripped so "A." counts as the one-character initial it is.
    ext_tokens = _strip_punctuation(ext_name).split()
    gt_tokens = _strip_punctuation(gt_name).split()
    if len(ext_tokens) >= len(gt_tokens):
        return True
    return all(
        len(gt_tok) < 2 or _token_covered(gt_tok, ext_tokens, threshold) for gt_tok in gt_tokens
    )


def _fuzzy_name_f1(
    ext_names: list[str],
    gt_names: list[str],
    threshold: float = 0.85,
    full_names: bool = False,
) -> float:
    """Greedy-bipartite fuzzy F1 over normalized name strings.

    ``full_names`` marks the inputs as whole printed names rather than family
    names: those are compared order-insensitively (``_name_similarity``) and
    must additionally clear the token-coverage guard (``_tokens_covered``).
    Both are scoped to full names on purpose — applied to the family-only
    diagnostic they would forgive exactly the given/family boundary
    disagreements it exists to expose.
    """
    if not ext_names and not gt_names:
        return 1.0
    if not ext_names or not gt_names:
        return 0.0
    pairs = []
    for gi, gt_name in enumerate(gt_names):
        for ei, ext_name in enumerate(ext_names):
            sim = (
                _name_similarity(ext_name, gt_name)
                if full_names
                else JaroWinkler.normalized_similarity(ext_name, gt_name)
            )
            if sim < threshold:
                continue
            if full_names and not _tokens_covered(ext_name, gt_name, threshold):
                continue
            pairs.append((sim, gi, ei))
    tp = _greedy_match_count(pairs)
    precision = tp / len(ext_names)
    recall = tp / len(gt_names)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _full_name(d: dict) -> str:
    return _normalize_name(f"{d.get('given', '')} {d.get('family', '')}".strip())


def _family_or_full(d: dict) -> str:
    """Family name, falling back to the full name when ``family`` is empty.

    An author with no ``family`` used to be dropped from both sides. That let a
    zero-author prediction score 1.0 against a gold record whose only author is
    stored family-less (``{given: "Director-General", family: ""}`` — an
    organizational byline), and let a prediction that put the whole name in
    ``given`` escape the precision denominator entirely.
    """
    family = _normalize_name(d.get("family") or "")
    return family or _normalize_name(f"{d.get('given', '')} {d.get('family', '')}".strip())


def authors_family_f1(extracted: list[dict], ground_truth: list[dict]) -> float:
    """F1 matching on family names with fuzzy matching (Jaro-Winkler >= 0.85).

    Uses globally-greedy bipartite matching on Jaro-Winkler similarity to handle
    diacritics, OCR errors, and transliteration variants. Jaro-Winkler is
    appropriate here (designed for person-name strings). Authors with an empty
    ``family`` fall back to their full name (see ``_family_or_full``) rather
    than being dropped.
    """
    ext_names = [n for n in (_family_or_full(d) for d in extracted) if n]
    gt_names = [n for n in (_family_or_full(d) for d in ground_truth) if n]
    return _fuzzy_name_f1(ext_names, gt_names)


def authors_fullname_f1(extracted: list[dict], ground_truth: list[dict]) -> float:
    """F1 over full printed names ("given family"), boundary-insensitive.

    The PDF prints a flat name string with no marker of where the family name
    begins, so a given/family split difference on an otherwise identical name
    (e.g. "Veras de Paula" vs "de Paula") is a naming-convention disagreement,
    not an extraction error. Same fuzzy greedy matching as authors_family_f1
    (order-insensitive, see ``_name_similarity``), plus a token-coverage guard
    so a shorter candidate cannot match on Jaro-Winkler's prefix boost alone
    (``_tokens_covered``).
    """
    ext_names = [n for n in (_full_name(d) for d in extracted) if n]
    gt_names = [n for n in (_full_name(d) for d in ground_truth) if n]
    return _fuzzy_name_f1(ext_names, gt_names, full_names=True)


def authors_count_ratio(extracted: list, ground_truth: list) -> float:
    """min(len(extracted), len(ground_truth)) / max(...), or 1.0 if both empty."""
    n_ext = len(extracted)
    n_gt = len(ground_truth)

    if n_ext == 0 and n_gt == 0:
        return 1.0
    if n_ext == 0 or n_gt == 0:
        return 0.0

    return min(n_ext, n_gt) / max(n_ext, n_gt)


def authors_order_score(extracted: list[dict], ground_truth: list[dict]) -> float:
    """Kendall's tau-like order correlation on matched authors (0-1).

    Matches authors by fuzzy family name, then measures how well the extraction
    preserves the ground-truth ordering. Returns 1.0 for perfect order,
    0.0 for fully reversed, 0.5 for random.
    """
    ext_names = [_normalize_name(d.get("family", "")) for d in extracted if d.get("family")]
    gt_names = [_normalize_name(d.get("family", "")) for d in ground_truth if d.get("family")]

    if len(ext_names) < 2 or len(gt_names) < 2:
        return 1.0

    threshold = 0.85
    pairs = []
    for gi, gt_name in enumerate(gt_names):
        for ei, ext_name in enumerate(ext_names):
            sim = JaroWinkler.normalized_similarity(ext_name, gt_name)
            if sim >= threshold:
                pairs.append((sim, gi, ei))

    matched = _greedy_match_pairs(pairs)

    if len(matched) < 2:
        return 1.0

    # Count concordant and discordant pairs
    n = len(matched)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_order = matched[i][0] - matched[j][0]
            ext_order = matched[i][1] - matched[j][1]
            if gt_order * ext_order > 0:
                concordant += 1
            elif gt_order * ext_order < 0:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 1.0

    # Normalize tau from [-1, 1] to [0, 1]
    tau = (concordant - discordant) / total_pairs
    return (tau + 1) / 2


# ---------------------------------------------------------------------------
# First-author metric
# ---------------------------------------------------------------------------


def first_author_match(extracted: list[dict], ground_truth: list[dict]) -> float:
    """Fuzzy similarity on the first author's family name (0-1)."""
    if not extracted and not ground_truth:
        return 1.0
    if not extracted or not ground_truth:
        return 0.0

    ext_first = (extracted[0].get("family") or "").strip()
    gt_first = (ground_truth[0].get("family") or "").strip()

    if not ext_first and not gt_first:
        return 1.0
    if not ext_first or not gt_first:
        return 0.0

    return token_sort_ratio(ext_first.lower(), gt_first.lower()) / 100.0


# ---------------------------------------------------------------------------
# Reference metrics
# ---------------------------------------------------------------------------


def references_count_ratio(extracted_count: int, ground_truth_count: int) -> float:
    """min/max ratio of reference counts, or 1.0 if both zero."""
    if extracted_count == 0 and ground_truth_count == 0:
        return 1.0
    if extracted_count == 0 or ground_truth_count == 0:
        return 0.0

    return min(extracted_count, ground_truth_count) / max(extracted_count, ground_truth_count)


# ---------------------------------------------------------------------------
# Reference matching helpers
# ---------------------------------------------------------------------------


def _get_ref_title(ref: dict) -> str:
    """Extract title from a reference dict, handling both bibr and Crossref key conventions."""
    return (ref.get("title") or ref.get("article-title") or ref.get("article_title") or "").strip()


def _get_ref_author(ref: dict) -> str:
    """Extract first-author surname from a reference dict."""
    author = (ref.get("author") or "").strip().lower()
    if not author:
        return ""
    return re.split(r"[,\s]", author)[0]


def _get_ref_year(ref: dict) -> str:
    """Extract publication year from a reference dict."""
    return str(ref.get("year") or ref.get("publication_year") or "").strip()


_AUTHOR_STOPWORDS = {"and", "et", "al", "jr", "sr", "ed", "eds"}


def _ref_surname_tokens(ref: dict) -> list[str]:
    """Surname-ish tokens from a reference's author field.

    Accepts ``authors_list`` (list of family names, GROBID style) or
    ``authors`` (bibr-style string ``"Caravolas, M., Lervåg, A., & Hulme, C"``).
    Both forms are normalized through the same pipeline (diacritic strip,
    punctuation strip, whitespace split), so multi-word surnames tokenize
    identically regardless of source — initials and connector words drop out.
    """
    names = ref.get("authors_list")
    if isinstance(names, list):
        raw = " ".join(str(n) for n in names if n)
    else:
        raw = ref.get("authors") if isinstance(ref.get("authors"), str) else ""
    if not raw or not raw.strip():
        return []
    text = _strip_punctuation(_normalize_unicode(raw.lower()))
    return [t for t in text.split() if len(t) > 1 and t not in _AUTHOR_STOPWORDS]


def _author_list_correct(ext_ref: dict, gt_ref: dict) -> bool:
    """True when every gold surname is (fuzzily) present in the extraction.

    Recall over gold surnames: extra extraction tokens (full given names from
    structured sources) never hurt; a dropped author fails the pair.
    """
    gt_tokens = _ref_surname_tokens(gt_ref)
    ext_tokens = _ref_surname_tokens(ext_ref)
    if not gt_tokens:
        return False
    if not ext_tokens:
        return False
    for gt_tok in gt_tokens:
        if not any(JaroWinkler.normalized_similarity(gt_tok, et) >= 0.9 for et in ext_tokens):
            return False
    return True


def _get_ref_container(ref: dict) -> str:
    return str(ref.get("container") or ref.get("journal") or "").strip()


def _get_ref_volume(ref: dict) -> str:
    return str(ref.get("volume") or "").strip()


_PAGE_RANGE_RE = re.compile(r"^\s*(?P<first>.+?)\s*[-–—]\s*(?P<last>.+?)\s*$")


def _split_page_range(value: str) -> tuple[str, str] | None:
    match = _PAGE_RANGE_RE.fullmatch(value)
    if match is None:
        return None
    first = match.group("first").strip()
    last = match.group("last").strip()
    return (first, last) if first and last else None


def _get_ref_pages(ref: dict) -> tuple[str, str]:
    first = str(ref.get("first_page") or "").strip()
    last = str(ref.get("last_page") or "").strip()
    return _split_page_range(first) or _split_page_range(last) or (first, last)


def _pages_correct(ext_ref: dict, gt_ref: dict) -> bool:
    """First page must match exactly; last page must match when gold has one."""
    gt_first, gt_last = _get_ref_pages(gt_ref)
    ext_first, ext_last = _get_ref_pages(ext_ref)
    if not gt_first:
        return False
    if ext_first != gt_first:
        return False
    return not (gt_last and ext_last != gt_last)


def _ref_similarity(ext_ref: dict, gt_ref: dict) -> float:
    """Compute similarity between two reference dicts using a cascade.

    Returns a score in [0, 1] following GROBID's citation signature approach:
      - 1.0 for DOI match
      - 0.9 for title match (fuzzy token_sort_ratio >= 85)
      - 0.7 for first-author surname + year match
      - 0.6 for match against unstructured citation string
      - 0.0 for no match
    """
    # Level 1: DOI match
    ext_doi = normalize_doi(ext_ref.get("doi") or ext_ref.get("DOI") or "")
    gt_doi = normalize_doi(gt_ref.get("doi") or gt_ref.get("DOI") or "")
    if ext_doi and gt_doi and ext_doi == gt_doi:
        return 1.0

    # Level 2: Title match (fuzzy)
    ext_title = _get_ref_title(ext_ref)
    gt_title = _get_ref_title(gt_ref)
    if ext_title and gt_title and token_sort_ratio(ext_title.lower(), gt_title.lower()) >= 85.0:
        return 0.9

    # Level 3: First-author surname + year match
    ext_surname = _get_ref_author(ext_ref)
    gt_surname = _get_ref_author(gt_ref)
    ext_year = _get_ref_year(ext_ref)
    gt_year = _get_ref_year(gt_ref)

    if (
        ext_surname
        and gt_surname
        and ext_year
        and gt_year
        and ext_year == gt_year
        and token_sort_ratio(ext_surname, gt_surname) >= 85.0
    ):
        return 0.7

    # Level 4: Match against unstructured citation string (common in Crossref)
    gt_unstructured = (gt_ref.get("unstructured") or "").lower()
    if gt_unstructured and len(gt_unstructured) > 20:
        matches = 0
        checks = 0
        if ext_surname:
            checks += 1
            if ext_surname in gt_unstructured:
                matches += 1
        if ext_year:
            checks += 1
            if ext_year in gt_unstructured:
                matches += 1
        if ext_title and len(ext_title) > 10:
            checks += 1
            title_words = ext_title.lower().split()
            word_matches = sum(1 for w in title_words if w in gt_unstructured)
            if word_matches >= len(title_words) * 0.6:
                matches += 1
        if checks >= 2 and matches >= 2:
            return 0.6

    return 0.0


def ref_matching_f1(extracted_refs: list[dict], ground_truth_refs: list[dict]) -> float:
    """F1 for individual reference matching using globally-greedy bipartite matching.

    Computes pairwise similarity using DOI > title > author+year > unstructured
    cascade, then matches using globally-greedy algorithm (picks highest-scoring
    pair first). This avoids the order-dependent artifacts of sequential greedy
    matching.
    """
    if not extracted_refs and not ground_truth_refs:
        return 1.0
    if not extracted_refs or not ground_truth_refs:
        return 0.0

    # Build similarity pairs above threshold
    pairs = []
    for gi, gt_ref in enumerate(ground_truth_refs):
        for ei, ext_ref in enumerate(extracted_refs):
            sim = _ref_similarity(ext_ref, gt_ref)
            if sim > 0:
                pairs.append((sim, gi, ei))

    tp = _greedy_match_count(pairs)

    precision = tp / len(extracted_refs)
    recall = tp / len(ground_truth_refs)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def ref_field_scores(
    extracted_refs: list[dict], ground_truth_refs: list[dict]
) -> dict[str, float | None]:
    """Per-field accuracy on matched reference pairs.

    Returns dict with keys:
    - ref_title_acc: among matched GT refs with titles, fraction with correct title (fuzzy >= 85)
    - ref_year_acc: among matched GT refs with years, fraction with correct year
    - ref_doi_recall: among ALL GT refs with DOIs, fraction extracted with correct DOI —
      an unmatched gold ref with a DOI counts as a miss
    - ref_author_acc: among matched GT refs with authors, fraction where all gold surnames
      are recalled in the extraction (JaroWinkler >= 0.9 per surname)
    - ref_journal_acc: among matched GT refs with a container/journal name, fraction correct
      (token_sort_ratio >= 85)
    - ref_volume_acc: among matched GT refs with a volume, fraction with exact volume match
    - ref_pages_acc: among matched GT refs with a first_page, fraction correct (first page
      exact; last page exact when gold carries one)

    A metric is None when the ground truth has no refs with that field
    (nothing to evaluate against, e.g. a paper that prints no reference
    DOIs) — callers must exclude None from aggregates rather than average
    it as 0.
    """
    gold_title_total = sum(1 for r in ground_truth_refs if _get_ref_title(r))
    gold_year_total = sum(1 for r in ground_truth_refs if _get_ref_year(r))
    gold_doi_total = sum(
        1 for r in ground_truth_refs if normalize_doi(r.get("doi") or r.get("DOI") or "")
    )
    gold_author_total = sum(1 for r in ground_truth_refs if _ref_surname_tokens(r))
    gold_journal_total = sum(1 for r in ground_truth_refs if _get_ref_container(r))
    gold_volume_total = sum(1 for r in ground_truth_refs if _get_ref_volume(r))
    gold_pages_total = sum(1 for r in ground_truth_refs if _get_ref_pages(r)[0])

    title_gt_matched = 0
    title_correct = 0
    year_gt_matched = 0
    year_correct = 0
    doi_correct = 0
    author_gt_matched = author_correct = 0
    journal_gt_matched = journal_correct = 0
    volume_gt_matched = volume_correct = 0
    pages_gt_matched = pages_correct_n = 0

    if extracted_refs and ground_truth_refs:
        pairs = []
        for gi, gt_ref in enumerate(ground_truth_refs):
            for ei, ext_ref in enumerate(extracted_refs):
                sim = _ref_similarity(ext_ref, gt_ref)
                if sim > 0:
                    pairs.append((sim, gi, ei))

        for gi, ei in _greedy_match_pairs(pairs):
            gt_ref = ground_truth_refs[gi]
            ext_ref = extracted_refs[ei]

            # Title accuracy (only count pairs where GT has a title)
            gt_title = _get_ref_title(gt_ref)
            if gt_title:
                title_gt_matched += 1
                ext_title = _get_ref_title(ext_ref)
                if ext_title and token_sort_ratio(ext_title.lower(), gt_title.lower()) >= 85.0:
                    title_correct += 1

            # Year accuracy (only count pairs where GT has a year)
            gt_year = _get_ref_year(gt_ref)
            if gt_year:
                year_gt_matched += 1
                ext_year = _get_ref_year(ext_ref)
                if ext_year and gt_year == ext_year:
                    year_correct += 1

            gt_doi = normalize_doi(gt_ref.get("doi") or gt_ref.get("DOI") or "")
            if gt_doi:
                ext_doi = normalize_doi(ext_ref.get("doi") or ext_ref.get("DOI") or "")
                if ext_doi and ext_doi == gt_doi:
                    doi_correct += 1

            if _ref_surname_tokens(gt_ref):
                author_gt_matched += 1
                if _author_list_correct(ext_ref, gt_ref):
                    author_correct += 1

            gt_container = _get_ref_container(gt_ref)
            if gt_container:
                journal_gt_matched += 1
                ext_container = _get_ref_container(ext_ref)
                if (
                    ext_container
                    and token_sort_ratio(ext_container.lower(), gt_container.lower()) >= 85.0
                ):
                    journal_correct += 1

            gt_volume = _get_ref_volume(gt_ref)
            if gt_volume:
                volume_gt_matched += 1
                if _get_ref_volume(ext_ref) == gt_volume:
                    volume_correct += 1

            if _get_ref_pages(gt_ref)[0]:
                pages_gt_matched += 1
                if _pages_correct(ext_ref, gt_ref):
                    pages_correct_n += 1

    def accuracy(correct: int, matched_total: int, gold_total: int) -> float | None:
        if gold_total == 0:
            return None
        # Gold has the field but no matched pair carries it → extraction miss.
        return correct / matched_total if matched_total else 0.0

    return {
        "ref_title_acc": accuracy(title_correct, title_gt_matched, gold_title_total),
        "ref_year_acc": accuracy(year_correct, year_gt_matched, gold_year_total),
        "ref_doi_recall": doi_correct / gold_doi_total if gold_doi_total else None,
        "ref_author_acc": accuracy(author_correct, author_gt_matched, gold_author_total),
        "ref_journal_acc": accuracy(journal_correct, journal_gt_matched, gold_journal_total),
        "ref_volume_acc": accuracy(volume_correct, volume_gt_matched, gold_volume_total),
        "ref_pages_acc": accuracy(pages_correct_n, pages_gt_matched, gold_pages_total),
    }


# ---------------------------------------------------------------------------
# Paper-level pass/fail floors
# ---------------------------------------------------------------------------
#
# When primary metrics approach 1.0, their means can hide papers that regress
# badly. FLOORS defines, per
# primary metric, the minimum score a single paper must clear; pass_rate()
# turns that into a paper-level aggregate (fraction of papers that clear every
# floor) that a saturated mean cannot mask.
#
# Scale and rationale for the defaults:
# - title_soft: 1.0/0.0 (exact after normalization) — 0.9 tolerates none of
#   its own noise, so it really means "must be an exact soft match".
# - doi_match: 1.0/0.0/None (None when the paper prints no DOI) — a printed
#   DOI must always be recovered verbatim; floor is 1.0 since anything else
#   is a wrong DOI, not a partial one.
# - authors_fullname_f1: fuzzy F1 on full printed names ("given family"),
#   boundary-insensitive — 0.9 tolerates at most one dropped/extra author on a
#   typical 5-8 author paper. The printed page carries no marker of where the
#   family name begins, so gating on family-only names would penalize a
#   given/family split disagreement on an otherwise byte-identical name as if
#   it were an extraction error. Family-only authors_f1 is still computed and
#   reported (see PRIMARY_METRIC_COLS in evaluate.py) as a diagnostic, but is
#   not part of the gate.
# - ref_matching_f1: bipartite-matched F1 over the reference list — 0.8
#   tolerates the segmentation/parsing noise floor without passing papers
#   that dropped a meaningful fraction of the bibliography.
# A paper whose value for a given floor metric is None (ground truth has
# nothing to score against for that paper) does not fail that floor — the
# metric is simply excluded, matching how it's excluded from mean aggregates.
FLOORS: dict[str, float] = {
    "title_soft": 0.9,
    "doi_match": 1.0,
    "authors_fullname_f1": 0.9,
    "ref_matching_f1": 0.8,
}


def paper_passes_floors(
    row: Mapping[str, float | None], floors: Mapping[str, float] = FLOORS
) -> bool:
    """True iff every floor metric present (non-None) in ``row`` meets its floor.

    A metric missing from ``row`` or holding ``None`` (nothing to score for
    that paper) never fails the paper — only a present value below its floor
    does.
    """
    for metric, floor in floors.items():
        value = row.get(metric)
        if value is None:
            continue
        if value < floor:
            return False
    return True


def pass_rate(
    rows: Iterable[Mapping[str, float | None]], floors: Mapping[str, float] = FLOORS
) -> float | None:
    """Fraction of ``rows`` (one dict of metrics per paper) that pass every floor.

    None when ``rows`` is empty — nothing to aggregate.
    """
    rows = list(rows)
    if not rows:
        return None
    passing = sum(1 for row in rows if paper_passes_floors(row, floors))
    return passing / len(rows)
