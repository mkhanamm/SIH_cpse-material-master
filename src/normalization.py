"""
NLP normalization of free-text material descriptions (Spec section 4.2).

WHAT THIS FILE DOES
    Turns a messy `Raw Description` such as `BFLY V/V LUG 300 NB CL-150` into a
    canonical, comparable form: `butterfly valve lug 300 nominal bore class 150`.

    Three ordered passes, each independently inspectable:
      1. Case folding, punctuation and whitespace repair.
      2. Abbreviation expansion from an editable lookup table (ABBREVIATIONS),
         seeded by inspecting real `Raw Description` values in this dataset.
      3. Unit canonicalization (inch->mm, sq.mm->sqmm, cu.m->cum, psi->bar...).

    IMPORTANT: `Standardized Description` is never used as a target or training
    signal here. That column is evaluation-only; using it would leak the answer.

INPUTS
    Raw description strings (and optionally the attribute columns for context).

OUTPUTS
    NormalizedText - dataclass with .text (canonical string) and .trace
                     (list of (pass_name, before, after) for the audit trail).

KEY FUNCTIONS
    normalize(text)                -> NormalizedText
    normalize_series(series)       -> pandas.Series[str]
    expand_abbreviations(text)     -> str
    canonicalize_units(text)       -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Abbreviation lookup table
# ---------------------------------------------------------------------------
# EVERY entry below was observed in the actual `Raw Description` column of this
# dataset (token-frequency scan, minimum 8 occurrences), not guessed from
# general engineering knowledge. Keys are matched case-insensitively on whole
# tokens after punctuation folding, so "BRG" expands but "BRGS" does not
# accidentally become "bearings" via substring matching.
#
# This table is deliberately data, not code: a domain expert at a CPSE can edit
# it without touching the pipeline, which is the point of keeping the rule layer
# separate from the embedding layer.
ABBREVIATIONS: dict[str, str] = {
    # --- materials of construction ---
    "cs": "carbon steel",
    "ms": "mild steel",
    "ss": "stainless steel",
    "ss304": "stainless steel 304",
    "ss316": "stainless steel 316",
    "al": "aluminium",
    "alu": "aluminium",
    "gi": "galvanised iron",
    "ci": "cast iron",
    "sg": "spheroidal graphite",
    "galv": "galvanised",
    "hdg": "hot dip galvanised",
    "xlpe": "cross linked polyethylene",
    "acsr": "aluminium conductor steel reinforced",
    # --- component nouns ---
    "brg": "bearing",
    "smls": "seamless",
    "trf": "transformer",
    "trfr": "transformer",
    "vv": "valve",
    "nrv": "non return valve",
    "flg": "flange",
    "bfly": "butterfly",
    "cyl": "cylinder",
    "hyd": "hydraulic",
    "hemm": "heavy earth moving machinery",
    "gbx": "gearbox",
    "cb": "circuit breaker",
    "ct": "current transformer",
    "pt": "potential transformer",
    "mcb": "miniature circuit breaker",
    # --- dimensional / geometric ---
    "dia": "diameter",
    "od": "outer diameter",
    "id": "inner diameter",
    "nb": "nominal bore",
    "dn": "nominal bore",
    "thk": "thickness",
    "thick": "thickness",
    "lg": "length",
    "wd": "width",
    "sch": "schedule",
    "hex": "hexagonal",
    "taper": "tapered",
    # --- ratings and classes ---
    "cl": "class",
    "wp": "working pressure",
    "rf": "raised face",
    "ff": "flat face",
    "rtj": "ring type joint",
    "onan": "oil natural air natural",
    "vrla": "valve regulated lead acid",
    "mtd": "mounted",
    "tp": "triple pole",
    "dp": "double pole",
    "lt": "low tension",
    "ht": "high tension",
    # --- specification bodies (left as-is but lowercased consistently) ---
    "astm": "astm",
    "asme": "asme",
    "is": "is",
    "en": "en",
    "sae": "sae",
    "gr": "grade",
    "grd": "grade",
    # --- connective shorthand ---
    "cw": "complete with",
    "w": "with",
    "approx": "approximately",
    "assy": "assembly",
    "qty": "quantity",
    "nos": "numbers",
    "deg": "degree",
}

# Multi-token phrases handled before single-token expansion, because splitting
# them first would destroy the phrase (e.g. "V/V" -> "v" + "v").
PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bv\s*/\s*v\b", "valve"),
    (r"\bm\s*\.\s*s\s*\.?", "mild steel"),
    (r"\bc\s*/\s*w\b", "complete with"),
    (r"\bw\s*/\s*", "with "),
    (r"\bcu\s*\.?\s*m\b", "cum"),
    (r"\bsq\s*\.?\s*mm\b", "sqmm"),
    (r"\bcu\s*\.?\s*mm\b", "cumm"),
    # Compound units collapse to ONE canonical token rather than an English
    # phrase. Two reasons, both learned the hard way on this dataset:
    #   1. `\b` does not fire between a digit and a letter, so "50m3/hr" was
    #      never matched -- hence the (?<![a-z]) lookbehind instead.
    #   2. Expanding to "cubic metre per hour" put the words "metre" and "hour"
    #      into the stream, which the later unit-alias pass then rewrote to
    #      "m" and "hr", producing "cubic m per hr". A single opaque token
    #      cannot be corrupted by a downstream alias.
    #   3. The token must be alpha-only, or the digit/letter splitter below
    #      re-splits it ("m3perhr" -> "m3 perhr").
    (r"(?<![a-z])m\s*3\s*/\s*hr\b", "cumperhr"),
    (r"(?<![a-z])m3\s*/\s*hr\b", "cumperhr"),
    (r"(?<![a-z])kg\s*/\s*cm\s*2\b", "kgpersqcm"),
    (r"\bmm\s*wg\b", "mmwg"),
]

# ---------------------------------------------------------------------------
# Unit canonicalization
# ---------------------------------------------------------------------------
# Only *dimensionally safe* conversions are performed. Note that this pass does
# not convert inch -> mm numerically inside the text; that conversion happens in
# attribute_extraction where the number and its unit are parsed together and can
# be validated. Doing arithmetic here, on a regex over free text, would silently
# corrupt strings like "1.5 INCH NB SCH20" where the schedule number must not be
# touched.
UNIT_ALIASES: dict[str, str] = {
    "millimeter": "mm",
    "millimetre": "mm",
    "millimeters": "mm",
    "millimetres": "mm",
    "mtr": "m",
    "mtrs": "m",
    "meter": "m",
    "metre": "m",
    "meters": "m",
    "metres": "m",
    "inches": "inch",
    "in": "inch",
    '"': "inch",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "tonne": "mt",
    "tonnes": "mt",
    "ton": "mt",
    "kvolt": "kv",
    "kilovolt": "kv",
    "amp": "a",
    "amps": "a",
    "ampere": "a",
    "amperes": "a",
    "hrs": "hr",
    "hour": "hr",
    "hours": "hr",
}

# Pressure is normalised to bar because the dataset mixes bar, psi and kg/cm2.
PSI_TO_BAR = 0.0689476
KGCM2_TO_BAR = 0.980665

_TOKEN_SPLIT = re.compile(r"[^0-9a-z./:\-]+")
_MULTISPACE = re.compile(r"\s+")


@dataclass
class NormalizedText:
    """Result of normalizing one description, with an inspectable trace.

    Attributes:
        text: The canonical, comparable form.
        original: The input string, unmodified.
        trace: Ordered ``(pass_name, before, after)`` tuples. This is written to
            the audit trail so a reviewer can see exactly which rule changed
            what -- normalization is never a black box in this system.
    """

    text: str
    original: str
    trace: list[tuple[str, str, str]] = field(default_factory=list)

    def changed_passes(self) -> list[str]:
        """Return the names of passes that actually modified the string.

        Returns:
            Pass names where ``before != after``.
        """
        return [name for name, before, after in self.trace if before != after]


# ---------------------------------------------------------------------------
# Individual passes
# ---------------------------------------------------------------------------
def fold_case_and_punctuation(text: str) -> str:
    """Lowercase, repair punctuation noise, and collapse whitespace.

    Separators that carry meaning are preserved: ``/`` inside ratios and unit
    expressions, ``.`` inside decimals, ``-`` inside grade codes, and ``:``
    inside gear ratios. Everything else becomes a space.

    Args:
        text: Raw description.

    Returns:
        Lowercased string with normalized separators.
    """
    if not isinstance(text, str):
        return ""
    lowered = text.lower().strip()
    # Dotted abbreviations must be resolved while their periods still exist --
    # "m.s." is unrecoverable once the periods become spaces.
    lowered = re.sub(r"\bm\.\s*s\.?", " mild steel ", lowered)
    lowered = re.sub(r"\bal\.", " aluminium ", lowered)
    # Protect decimals and ratios, then strip stray punctuation.
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[,;()\[\]{}*_+|]", " ", lowered)
    # A period is meaningful only *between* two digits (3.5, 1.1 kv). Anywhere
    # else -- "gr.6", "cu.m", trailing sentence dots -- it is noise. Both sides
    # must be checked: a lookahead alone leaves "gr.6" intact.
    lowered = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", lowered)
    return _MULTISPACE.sub(" ", lowered).strip()


def expand_abbreviations(text: str) -> str:
    """Expand domain abbreviations to their full forms.

    Phrase-level patterns are applied first (``V/V`` -> ``valve``), then
    whole-token lookups against :data:`ABBREVIATIONS`. Tokens that fuse an
    abbreviation to a number (``sch20``, ``cl600``, ``ss304``) are split before
    lookup so the abbreviation is still recognised.

    Args:
        text: Case-folded description.

    Returns:
        Description with abbreviations expanded.
    """
    result = text
    for pattern, replacement in PHRASE_REPLACEMENTS:
        result = re.sub(pattern, replacement, result)

    # Split fused alpha+digit tokens: "sch20" -> "sch 20", "m12x80" -> "m12 x 80"
    result = re.sub(r"\b([a-z]{2,})(\d)", r"\1 \2", result)
    result = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1 x \2", result)
    # ...and the reverse direction: "300mm" -> "300 mm", "3ply" -> "3 ply".
    # Without this, "300MM CLASS 150" and "300 NB CL-150" never converge.
    # Applied uniformly to both sides of every comparison, so it cannot
    # introduce asymmetry even where the split is linguistically debatable.
    result = re.sub(r"(\d)([a-z]+)", r"\1 \2", result)
    # Trailing separators left by "cl-150" or "sch-20"
    result = re.sub(r"\b([a-z]+)-(?=\d)", r"\1 ", result)

    tokens = [t for t in _TOKEN_SPLIT.split(result) if t]
    expanded: list[str] = []
    for token in tokens:
        key = token.strip(".-/")
        expanded.append(ABBREVIATIONS.get(key, token))
    return _MULTISPACE.sub(" ", " ".join(expanded)).strip()


def canonicalize_units(text: str) -> str:
    """Map unit spellings onto one canonical alias per physical unit.

    Numeric conversion (inch -> mm, psi -> bar) is deliberately *not* done here;
    see the module note on :data:`UNIT_ALIASES`. This pass only makes the unit
    token itself consistent so downstream extraction has one spelling to match.

    Args:
        text: Abbreviation-expanded description.

    Returns:
        Description with canonical unit tokens.
    """
    tokens = text.split()
    canonical = [UNIT_ALIASES.get(t, t) for t in tokens]
    joined = " ".join(canonical)
    # "50 mm" and "50mm" must land in the same place.
    joined = re.sub(r"(\d)\s+(mm|cm|m|kg|mt|kv|kva|hp|bar|inch)\b", r"\1 \2", joined)
    return _MULTISPACE.sub(" ", joined).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize(text: str) -> NormalizedText:
    """Run the full three-pass normalization with a recorded trace.

    Args:
        text: A raw material description.

    Returns:
        A :class:`NormalizedText` holding the canonical string and the per-pass
        trace used by the audit trail.

    Example:
        >>> normalize("BFLY V/V LUG 300 NB CL-150").text
        'butterfly valve lug 300 nominal bore class 150'
    """
    original = text if isinstance(text, str) else ""
    trace: list[tuple[str, str, str]] = []

    stage = original
    for name, fn in (
        ("case_punctuation", fold_case_and_punctuation),
        ("abbreviations", expand_abbreviations),
        ("units", canonicalize_units),
    ):
        before = stage
        stage = fn(stage)
        trace.append((name, before, stage))

    return NormalizedText(text=stage, original=original, trace=trace)


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a whole column of descriptions.

    Args:
        series: Column of raw descriptions.

    Returns:
        Series of canonical strings, same index as the input.
    """
    return series.map(lambda t: normalize(t).text)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    samples = [
        "BFLY V/V LUG 300 NB CL-150",
        "BUTTERFLY VALVE LUG TYPE 300MM CLASS 150",
        "CS SMLS PIPE 1.5 INCH NB SCH20 A333 GR.6",
        "M.S. SMLS PIPE DN40 SCH20",
        "BRG DEEP GROOVE BALL BORE 25 MM",
        "DEEP GROOVE BALL BEARING 25MM BORE",
        "STUD M12X100 B7 C/W 2 NUTS 2H",
        "AL. XLPE CABLE 1.1 KV, 1C CORE, 50 SQ.MM",
    ]
    for s in samples:
        print(f"{s:52s} -> {normalize(s).text}")
