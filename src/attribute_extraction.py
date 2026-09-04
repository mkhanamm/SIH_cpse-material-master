"""
Rule-based structured attribute extraction (Spec section 4.3).

WHAT THIS FILE DOES
    Pulls structured technical attributes out of normalized description text
    using deterministic, inspectable regex rules -- deliberately NOT a model, so
    a reviewer can always see exactly why a value was extracted.

    Extracted attributes: nominal_size_mm, thickness_mm, grade, spec_standard,
    schedule, pressure_class, capacity, rating_kva, voltage_kv, ratio, ply,
    material_of_construction.

    Where the dataset's own semi-structured columns (`Material/Grade`,
    `Dimensions`, `Specification/Standard`, `Capacity/Rating`,
    `Operating Parameter`) are populated, they are merged in and used to
    cross-validate the regex output (see `validate_against_columns`).

    Missing values degrade gracefully to None and are reported as
    "attribute unknown" -- never fabricated. Downstream, an unknown attribute
    contributes neutral evidence, not a match.

INPUTS
    Normalized description text + the row's semi-structured attribute columns.

OUTPUTS
    AttributeSet - dataclass of typed optional attributes plus .provenance
                   ({attribute: 'regex' | 'column' | 'both' | 'unknown'}).

KEY FUNCTIONS
    extract(text, row)                    -> AttributeSet
    extract_frame(df, text_col)           -> pandas.DataFrame
    validate_against_columns(df)          -> dict  (agreement rates, for the report)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import config
from .normalization import normalize

# ---------------------------------------------------------------------------
# Nominal bore conversion table
# ---------------------------------------------------------------------------
# CRITICAL DOMAIN RULE: nominal bore in inches does NOT convert to millimetres
# by arithmetic. A 1.5-inch NB pipe is designated 40 mm NB, not 38.1 mm. The
# dataset confirms this -- `CS SMLS PIPE 1.5 INCH NB SCH20` has the standardized
# form `40 mm NB`. Multiplying by 25.4 would place it 5% away from every 40 mm
# record and the match would be missed under any sane tolerance.
#
# This table is the single highest-value piece of domain knowledge in the
# extraction layer, and it is exactly the kind of rule a pure-embedding approach
# cannot represent.
INCH_TO_NB_MM: dict[float, float] = {
    0.5: 15.0, 0.75: 20.0, 1.0: 25.0, 1.25: 32.0, 1.5: 40.0, 2.0: 50.0,
    2.5: 65.0, 3.0: 80.0, 3.5: 90.0, 4.0: 100.0, 5.0: 125.0, 6.0: 150.0,
    8.0: 200.0, 10.0: 250.0, 12.0: 300.0, 14.0: 350.0, 16.0: 400.0,
    18.0: 450.0, 20.0: 500.0, 24.0: 600.0, 30.0: 750.0, 36.0: 900.0,
}

# Material-of-construction keywords, checked against normalized (expanded) text.
MATERIAL_KEYWORDS: dict[str, str] = {
    "carbon steel": "carbon steel",
    "mild steel": "carbon steel",  # MS and CS are the same procurement class
    "stainless steel 304": "stainless steel 304",
    "stainless steel 316": "stainless steel 316",
    "stainless steel": "stainless steel",
    "alloy steel": "alloy steel",
    "cast iron": "cast iron",
    "spheroidal graphite": "spheroidal graphite iron",
    "aluminium": "aluminium",
    "copper": "copper",
    "rubber": "rubber",
    "hardox": "hardox",
    "manganese": "manganese steel",
}

_NUM = r"(\d+(?:\.\d+)?)"

# Each pattern is a (name, compiled_regex, group_index) triple. Order matters
# only where patterns could compete for the same token; more specific first.
PATTERNS: dict[str, re.Pattern[str]] = {
    # "40 nominal bore", "300 mm nominal bore", "nominal bore 40"
    "nominal_size_mm": re.compile(
        rf"(?:{_NUM}\s*(?:mm\s*)?nominal bore|nominal bore\s*{_NUM}\s*(?:mm)?)"
    ),
    "inch_size": re.compile(rf"{_NUM}\s*inch"),
    "bore_mm": re.compile(rf"{_NUM}\s*mm\s*bore|bore\s*{_NUM}\s*mm"),
    "thickness_mm": re.compile(rf"{_NUM}\s*mm\s*thickness|thickness\s*{_NUM}\s*mm"),
    "schedule": re.compile(r"schedule\s*(\d+\s*s?|std|xs|xxs)"),
    "pressure_class": re.compile(r"class\s*(\d{2,4})"),
    "spec_standard": re.compile(
        r"\b(astm\s*a\s*\d+|asme\s*b?\s*\d+(?:\.\d+)?|is\s*\d{3,4}|en\s*\d{3,5}|"
        r"api\s*\d+[a-z]?|din\s*\d+|sae\s*\d+)\b"
    ),
    "grade": re.compile(
        r"\bgrade\s*([a-z0-9]+)\b|\b(e\s?\d{3})\b|\bb\s?7\b|\bwpb\b|\bgr\s*([a-z0-9]+)\b"
    ),
    "rating_kva": re.compile(rf"{_NUM}\s*kva"),
    "voltage_kv": re.compile(rf"{_NUM}\s*kv"),
    "power_hp": re.compile(rf"{_NUM}\s*hp"),
    "flow_m3hr": re.compile(rf"{_NUM}\s*cumperhr"),
    "head_m": re.compile(rf"(?:head\s*{_NUM}\s*m\b|{_NUM}\s*m\s*head)"),
    # (?![a-z]) stops this matching the "cum" prefix of "cumperhr" -- a 50 m3/hr
    # flow rate is not a 50 cubic-metre capacity.
    "capacity_cum": re.compile(rf"{_NUM}\s*cum(?![a-z])"),
    "area_sqmm": re.compile(rf"{_NUM}\s*sqmm"),
    "ratio": re.compile(r"(\d+)\s*:\s*(\d+)"),
    "ply": re.compile(r"(\d+)\s*ply"),
    "width_mm": re.compile(rf"{_NUM}\s*mm\s*(?:width|wide)|width\s*{_NUM}\s*mm"),
}

# Attributes whose value is numeric and therefore compared with a tolerance.
NUMERIC_ATTRIBUTES = [
    "nominal_size_mm", "thickness_mm", "rating_kva", "voltage_kv", "power_hp",
    "flow_m3hr", "head_m", "capacity_cum", "area_sqmm", "ply", "width_mm",
]
# Attributes compared for exact equality -- see config.EXACT_MATCH_ATTRIBUTES.
CATEGORICAL_ATTRIBUTES = [
    "grade", "spec_standard", "schedule", "pressure_class", "ratio",
    "material_of_construction",
]
ALL_ATTRIBUTES = NUMERIC_ATTRIBUTES + CATEGORICAL_ATTRIBUTES


@dataclass
class AttributeSet:
    """Structured attributes extracted from one material record.

    Every field is Optional. ``None`` means "not stated in the source" and is
    treated downstream as neutral evidence -- never as a mismatch, and never
    filled with a guessed default.

    Attributes:
        provenance: Per-attribute origin -- ``'regex'`` (from description text),
            ``'column'`` (from the semi-structured column), ``'both'`` (agreeing
            sources), ``'conflict'`` (sources disagree; column wins and the
            disagreement is recorded), or ``'unknown'``.
    """

    nominal_size_mm: float | None = None
    thickness_mm: float | None = None
    rating_kva: float | None = None
    voltage_kv: float | None = None
    power_hp: float | None = None
    flow_m3hr: float | None = None
    head_m: float | None = None
    capacity_cum: float | None = None
    area_sqmm: float | None = None
    ply: float | None = None
    width_mm: float | None = None
    grade: str | None = None
    spec_standard: str | None = None
    schedule: str | None = None
    pressure_class: str | None = None
    ratio: str | None = None
    material_of_construction: str | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def known(self) -> dict[str, object]:
        """Return only the attributes that actually have a value.

        Returns:
            Mapping of attribute name to value, excluding ``None`` and the
            provenance dict.
        """
        return {
            k: v
            for k, v in asdict(self).items()
            if k != "provenance" and v is not None
        }

    def n_known(self) -> int:
        """Count populated attributes.

        Used by the confidence router: a pair with too few jointly-known
        attributes cannot be auto-approved, however similar its text.

        Returns:
            Number of non-None attributes.
        """
        return len(self.known())


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def _first_number(match: re.Match[str] | None) -> float | None:
    """Return the first non-empty numeric capture group of a match.

    Args:
        match: A regex match, or None.

    Returns:
        The captured number as a float, or None if there was no match.
    """
    if match is None:
        return None
    for group in match.groups():
        if group:
            try:
                return float(group)
            except ValueError:
                continue
    return None


def _first_string(match: re.Match[str] | None) -> str | None:
    """Return the first non-empty capture group of a match, whitespace-squashed.

    Args:
        match: A regex match, or None.

    Returns:
        The captured string, or None.
    """
    if match is None:
        return None
    for group in match.groups():
        if group:
            return re.sub(r"\s+", " ", group).strip()
    # Zero-group patterns (e.g. the bare "wpb" alternative) return the whole hit.
    return re.sub(r"\s+", " ", match.group(0)).strip() or None


def inch_to_nb_mm(inches: float) -> float | None:
    """Convert a nominal-bore size in inches to its millimetre designation.

    Uses :data:`INCH_TO_NB_MM` rather than multiplying by 25.4, because nominal
    bore is a designation, not a measurement.

    Args:
        inches: Size in inches as written in the description.

    Returns:
        The mm designation, or None if the size is not a standard NB value (in
        which case no value is invented).

    Example:
        >>> inch_to_nb_mm(1.5)
        40.0
    """
    return INCH_TO_NB_MM.get(round(inches, 2))


def _extract_from_text(text: str) -> tuple[dict[str, object], dict[str, str]]:
    """Apply every regex rule to normalized text.

    Args:
        text: Normalized description.

    Returns:
        A ``(values, provenance)`` pair; provenance entries are all ``'regex'``.
    """
    values: dict[str, object] = {}

    size = _first_number(PATTERNS["nominal_size_mm"].search(text))
    if size is None:
        size = _first_number(PATTERNS["bore_mm"].search(text))
    if size is None:
        inches = _first_number(PATTERNS["inch_size"].search(text))
        if inches is not None:
            size = inch_to_nb_mm(inches)
    if size is not None:
        values["nominal_size_mm"] = size

    for name in ("thickness_mm", "rating_kva", "voltage_kv", "power_hp",
                 "flow_m3hr", "head_m", "capacity_cum", "area_sqmm", "width_mm"):
        val = _first_number(PATTERNS[name].search(text))
        if val is not None:
            values[name] = val

    ply = PATTERNS["ply"].search(text)
    if ply:
        values["ply"] = float(ply.group(1))

    for name in ("schedule", "pressure_class", "spec_standard", "grade"):
        val = _first_string(PATTERNS[name].search(text))
        if val is not None:
            values[name] = re.sub(r"\s+", "", val) if name != "spec_standard" else val

    ratio = PATTERNS["ratio"].search(text)
    if ratio:
        values["ratio"] = f"{ratio.group(1)}:{ratio.group(2)}"

    for keyword, canonical in MATERIAL_KEYWORDS.items():
        if keyword in text:
            values["material_of_construction"] = canonical
            break

    return values, {k: "regex" for k in values}


def _extract_from_columns(row: pd.Series) -> tuple[dict[str, object], dict[str, str]]:
    """Read attributes out of the dataset's semi-structured columns.

    These columns are heavily null (Specification/Standard is absent in 4,342 of
    5,008 rows), so this returns whatever is present and nothing more.

    Args:
        row: A row of the pipeline frame.

    Returns:
        A ``(values, provenance)`` pair; provenance entries are all ``'column'``.
    """
    values: dict[str, object] = {}

    grade = row.get("Material/Grade")
    if isinstance(grade, str) and grade.strip():
        values["grade"] = grade.strip().lower()

    spec = row.get("Specification/Standard")
    if isinstance(spec, str) and spec.strip():
        values["spec_standard"] = spec.strip().lower()

    dims = row.get("Dimensions")
    if isinstance(dims, str) and dims.strip():
        norm_dims = normalize(dims).text
        size = _first_number(PATTERNS["nominal_size_mm"].search(norm_dims))
        if size is None:
            size = _first_number(re.search(rf"{_NUM}\s*mm", norm_dims))
        if size is not None:
            values["nominal_size_mm"] = size
        sched = _first_string(PATTERNS["schedule"].search(norm_dims))
        if sched:
            values["schedule"] = re.sub(r"\s+", "", sched)

    capacity = row.get("Capacity/Rating")
    if isinstance(capacity, str) and capacity.strip():
        norm_cap = normalize(capacity).text
        for name in ("rating_kva", "voltage_kv", "power_hp", "capacity_cum",
                     "flow_m3hr", "head_m"):
            val = _first_number(PATTERNS[name].search(norm_cap))
            if val is not None:
                values.setdefault(name, val)

    return values, {k: "column" for k in values}


def extract(text: str, row: pd.Series | None = None) -> AttributeSet:
    """Extract structured attributes from a description and its columns.

    Regex output and column output are merged. Where both sources supply a
    value, the column wins (it is the more curated source) but the agreement or
    disagreement is recorded in ``provenance`` so the explanation layer can show
    a reviewer that two sources concurred -- which is stronger evidence than
    either alone.

    Args:
        text: Raw or normalized description. Raw text is normalized internally,
            so callers cannot accidentally pass unnormalized input.
        row: Optional row of the pipeline frame carrying the semi-structured
            attribute columns.

    Returns:
        A populated :class:`AttributeSet`. Absent attributes stay ``None``.
    """
    normalized = normalize(text).text
    text_values, text_prov = _extract_from_text(normalized)

    col_values: dict[str, object] = {}
    col_prov: dict[str, str] = {}
    if row is not None:
        col_values, col_prov = _extract_from_columns(row)

    merged: dict[str, object] = dict(text_values)
    provenance: dict[str, str] = dict(text_prov)

    for key, value in col_values.items():
        if key in merged:
            same = _values_agree(merged[key], value)
            provenance[key] = "both" if same else "conflict"
            merged[key] = value  # curated column is authoritative
        else:
            merged[key] = value
            provenance[key] = col_prov[key]

    for attribute in ALL_ATTRIBUTES:
        provenance.setdefault(attribute, "unknown")

    return AttributeSet(provenance=provenance, **merged)  # type: ignore[arg-type]


def _values_agree(a: object, b: object) -> bool:
    """Test whether two extracted values represent the same thing.

    Numbers agree within ``config.DIMENSION_TOLERANCE_PCT``; strings agree if
    one contains the other after case folding (so ``astm a106`` matches
    ``astm a106 grade b``).

    Args:
        a: First value.
        b: Second value.

    Returns:
        True when the values are compatible.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0 or b == 0:
            return a == b
        return abs(a - b) / max(abs(a), abs(b)) <= config.DIMENSION_TOLERANCE_PCT
    sa, sb = str(a).lower().strip(), str(b).lower().strip()
    return sa == sb or sa in sb or sb in sa


# ---------------------------------------------------------------------------
# Frame-level API
# ---------------------------------------------------------------------------
def extract_frame(
    df: pd.DataFrame, text_col: str = config.INPUT_TEXT_COLUMN
) -> pd.DataFrame:
    """Extract attributes for every row of a material master.

    Args:
        df: Pipeline frame.
        text_col: Column holding the description text.

    Returns:
        A frame indexed like ``df`` with one column per attribute, plus
        ``attr_provenance`` (dict) and ``n_known_attributes`` (int).
    """
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        attrs = extract(row[text_col], row)
        record = attrs.known()
        record["attr_provenance"] = attrs.provenance
        record["n_known_attributes"] = attrs.n_known()
        records.append(record)

    out = pd.DataFrame(records, index=df.index)
    for attribute in ALL_ATTRIBUTES:
        if attribute not in out.columns:
            out[attribute] = None
    return out


def validate_against_columns(
    df: pd.DataFrame, extracted: pd.DataFrame
) -> dict[str, dict[str, float | int]]:
    """Cross-check regex extraction against the dataset's own attribute columns.

    This is the honesty check on the rule layer: where the dataset already
    states a grade or a dimension, does the regex reading of the free text
    agree? A low agreement rate means the rules are wrong, not that the data is.

    Args:
        df: Pipeline frame with the semi-structured columns.
        extracted: Output of :func:`extract_frame`.

    Returns:
        Per-attribute dict with ``n_comparable``, ``n_agree`` and
        ``agreement_rate``.
    """
    report: dict[str, dict[str, float | int]] = {}

    checks = [
        ("grade", "Material/Grade"),
        ("spec_standard", "Specification/Standard"),
        ("nominal_size_mm", "Dimensions"),
    ]
    for attribute, column in checks:
        if column not in df.columns:
            continue
        comparable = 0
        agree = 0
        for idx in df.index:
            raw_col = df.at[idx, column]
            if not isinstance(raw_col, str) or not raw_col.strip():
                continue
            # Re-extract from the description text alone, ignoring the column,
            # so the comparison is genuinely independent.
            text_only, _ = _extract_from_text(
                normalize(df.at[idx, config.INPUT_TEXT_COLUMN]).text
            )
            if attribute not in text_only:
                continue
            comparable += 1
            col_values, _ = _extract_from_columns(df.loc[idx])
            if attribute in col_values and _values_agree(
                text_only[attribute], col_values[attribute]
            ):
                agree += 1
        report[attribute] = {
            "n_comparable": comparable,
            "n_agree": agree,
            "agreement_rate": (agree / comparable) if comparable else 0.0,
        }
    return report


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from . import ingestion

    dataset = ingestion.load_dataset()
    sample = dataset.pipeline_df.head(8)
    for _, r in sample.iterrows():
        a = extract(r[config.INPUT_TEXT_COLUMN], r)
        print(f"{r[config.INPUT_TEXT_COLUMN][:50]:52s} -> {a.known()}")
