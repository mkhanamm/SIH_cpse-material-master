"""
Upload ingestion for the "Load Data" view (Spec section 3.5, view 1).

WHAT THIS FILE DOES
    Turns an arbitrary uploaded .xlsx/.csv into an `ingestion.MaterialDataset`
    that is otherwise indistinguishable from one loaded from the bundled demo
    workbook: same required column names, same `has_labels=False` contract
    (an upload never carries `GroundTruth_Group`), same profiling.

    Column mapping is explicit rather than positional: an uploaded file's
    headers rarely match the demo schema, so the caller (app.py) shows a
    preview and lets a human confirm which uploaded column plays which role.
    `suggest_column_mapping` only pre-fills that choice for headers that
    already match; it never guesses past what the UI presents for
    confirmation.

    A few columns the downstream pipeline reads unconditionally --
    `UOM` (profiling) and `Legacy_Sector_Code` (CNMC generation) -- are not
    part of the user-facing mapping (the task only calls out CPSE, CPSE
    Material Code, Material Category, Raw Description as required and
    Sector plus the five attribute columns as optional); they are filled
    with a harmless blank default instead, since nothing in the matching
    logic depends on their content.

INPUTS
    An uploaded file-like object (Streamlit's UploadedFile, or any buffer
    pandas can read) plus a user-confirmed column mapping.

OUTPUTS
    ingestion.MaterialDataset

KEY FUNCTIONS
    read_upload(file, filename)             -> pd.DataFrame
    suggest_column_mapping(columns)         -> dict[str, str | None]
    guess_column_mapping(df)                -> dict[str, str | None]
    infer_column_mapping(df)                -> dict[str, str | None]
    duplicate_required_sources(mapping)     -> dict[str, list[str]]
    apply_column_mapping(raw_df, mapping)   -> pd.DataFrame
    preview_mapped_row(raw_df, mapping)     -> str
    build_dataset_from_mapped(df, label)    -> MaterialDataset
    estimate_runtime_seconds(n_rows)        -> float
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config, ingestion

# Columns the "Load Data" view asks the user to map.
REQUIRED_COLUMNS = [
    "CPSE",
    "CPSE Material Code",
    "Material Category",
    config.INPUT_TEXT_COLUMN,
]
OPTIONAL_COLUMNS = ["Sector", *config.ATTRIBUTE_COLUMNS]
ALL_MAPPABLE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Columns the pipeline reads unconditionally but that are not part of the
# user-facing mapping -- see module docstring. Blank/empty rather than NaN,
# so downstream `str(...)` calls (cnmc_generator) never print "nan"/"<NA>".
_AUTO_DEFAULTED_COLUMNS = {"UOM": "", "Legacy_Sector_Code": ""}


class UnsupportedFileType(ValueError):
    """Raised when an uploaded file is neither .xlsx nor .csv."""


def read_upload(file, filename: str) -> pd.DataFrame:
    """Read an uploaded workbook or CSV into a raw DataFrame, unmodified.

    Args:
        file: File-like object (e.g. Streamlit's ``UploadedFile``) or a path.
        filename: Original filename, used only to pick the reader by
            extension -- an opened file-like object has no extension of its
            own.

    Returns:
        The first sheet (xlsx) or the whole file (csv).

    Raises:
        UnsupportedFileType: If the extension is neither .xlsx nor .csv.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(file)
    if suffix == ".csv":
        return pd.read_csv(file)
    raise UnsupportedFileType(
        f"Unsupported file type {suffix!r}: expected .xlsx or .csv."
    )


def _normalize_header(name: object) -> str:
    """Collapse whitespace and case so headers compare loosely but exactly.

    Args:
        name: A column header.

    Returns:
        Lowercased, whitespace-collapsed form for equality comparison.
    """
    return " ".join(str(name).split()).strip().lower()


def suggest_column_mapping(columns: list[str]) -> dict[str, str | None]:
    """Pre-fill a column mapping for headers that already match by name.

    This only matches on the header text itself (case/whitespace-insensitive
    exact match) -- it does not guess at synonyms or abbreviations. Anything
    it can't match is left for the user to pick manually in the UI.

    Args:
        columns: Column names from the uploaded file.

    Returns:
        ``{canonical_name: matched_source_column_or_None}`` for every
        required and optional column.
    """
    normalized = {_normalize_header(c): c for c in columns}
    return {
        target: normalized.get(_normalize_header(target))
        for target in ALL_MAPPABLE_COLUMNS
    }


# ---------------------------------------------------------------------------
# Data-shape guessing for the mapping UI
# ---------------------------------------------------------------------------
# Header names on a real CPSE export rarely match this schema, so a
# header-only match (suggest_column_mapping) leaves a first-time user staring
# at four empty dropdowns. These helpers inspect the *values* instead and
# pre-select a best guess, which the user can still override.

# "NTPC-VLV-2201", "M/12/0098" -- alphanumeric run(s) joined by - or /.
_CODE_LIKE = re.compile(r"^[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+$")
# "NTPC", "Oil & Gas", "Gate Valve (lugged)" -- a label, not free text or a code.
_NAME_LIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,&'()-]*$")

# Fields guess_column_mapping can infer (the four required ones).
_GUESSABLE_FIELDS = (
    "CPSE",
    "CPSE Material Code",
    "Material Category",
    config.INPUT_TEXT_COLUMN,
)


def _clean_values(series: pd.Series) -> list[str]:
    """Non-empty, non-null string values of a column, trimmed."""
    cleaned: list[str] = []
    for value in series.tolist():
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>", "null"}:
            continue
        cleaned.append(text)
    return cleaned


def column_hint(series: pd.Series, n_examples: int = 3, max_chars: int = 24) -> str:
    """A few distinct example values from a column, for the mapping dropdowns.

    A first-time user recognises ``"NTPC, SAIL, BHEL..."`` far faster than
    they match a bare header to the word "CPSE". Long values (descriptions)
    are truncated so the option label stays on one line.

    Args:
        series: The source column.
        n_examples: How many distinct values to show.
        max_chars: Truncate any single example longer than this.

    Returns:
        A comma-separated preview like ``"NTPC, SAIL, BHEL..."``; the empty
        string for a column with no usable values.
    """
    distinct: list[str] = []
    for text in _clean_values(series):
        if text not in distinct:
            distinct.append(text)
        if len(distinct) >= n_examples:
            break
    if not distinct:
        return ""
    shown = [
        value if len(value) <= max_chars else value[: max_chars - 1].rstrip() + "…"
        for value in distinct
    ]
    return ", ".join(shown) + ("..." if len(distinct) >= n_examples else "")


@dataclass
class _ColumnStats:
    """Cheap value-shape statistics for one uploaded column."""

    name: str
    count: int
    n_unique: int
    unique_ratio: float
    avg_len: float
    frac_code_like: float
    frac_name_like: float


def _column_stats(name: str, series: pd.Series) -> _ColumnStats | None:
    """Profile a column's values, or None if it has no usable values."""
    values = _clean_values(series)
    if not values:
        return None
    count = len(values)
    n_unique = len(set(values))
    avg_len = sum(len(v) for v in values) / count
    frac_code_like = sum(1 for v in values if _CODE_LIKE.match(v)) / count
    frac_name_like = sum(
        1 for v in values if len(v) <= 28 and _NAME_LIKE.match(v)
    ) / count
    return _ColumnStats(
        name=name,
        count=count,
        n_unique=n_unique,
        unique_ratio=n_unique / count,
        avg_len=avg_len,
        frac_code_like=frac_code_like,
        frac_name_like=frac_name_like,
    )


def guess_column_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    """Guess the four required mappings by inspecting column values.

    Heuristics, applied in order so an unambiguous column is claimed before a
    weaker signal can take it:

    1. **Raw Description** -- the column with the longest average text
       (>= 20 chars), which no code or label reaches.
    2. **CPSE Material Code** -- nearly every value distinct
       (unique ratio >= 0.9) and alphanumeric-with-separators more often
       than not.
    3. **CPSE** and **Material Category** -- both are short repeated labels;
       the company column has *fewer* distinct values than the category
       column, so the remaining label-like columns are ranked by distinct
       count and the smallest is taken for CPSE, the next for Category.

    Args:
        df: The uploaded file, unmodified.

    Returns:
        ``{field: source_column_or_None}`` for each of the four required
        fields. A field is left ``None`` when nothing matches confidently;
        the user still confirms every choice in the UI.
    """
    mapping: dict[str, str | None] = {field: None for field in _GUESSABLE_FIELDS}
    stats = [s for s in (_column_stats(c, df[c]) for c in df.columns) if s]
    if not stats:
        return mapping
    taken: set[str] = set()

    text_cols = sorted(
        (s for s in stats if s.avg_len >= 20), key=lambda s: s.avg_len, reverse=True
    )
    if text_cols:
        mapping[config.INPUT_TEXT_COLUMN] = text_cols[0].name
        taken.add(text_cols[0].name)

    code_cols = sorted(
        (
            s for s in stats
            if s.name not in taken
            and s.unique_ratio >= 0.9
            and s.frac_code_like >= 0.5
        ),
        key=lambda s: (s.frac_code_like, s.unique_ratio),
        reverse=True,
    )
    if code_cols:
        mapping["CPSE Material Code"] = code_cols[0].name
        taken.add(code_cols[0].name)

    label_cols = sorted(
        (
            s for s in stats
            if s.name not in taken
            and s.n_unique >= 2
            and s.unique_ratio <= 0.6
            and s.avg_len <= 30
            and s.frac_name_like >= 0.6
        ),
        key=lambda s: s.n_unique,
    )
    if label_cols:
        mapping["CPSE"] = label_cols[0].name
        taken.add(label_cols[0].name)
    if len(label_cols) >= 2:
        mapping["Material Category"] = label_cols[1].name
        taken.add(label_cols[1].name)

    return mapping


def infer_column_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    """Pre-fill the mapping from header names first, then from data shape.

    :func:`suggest_column_mapping` handles the easy case (a header already
    called ``CPSE``); :func:`guess_column_mapping` fills the required fields
    it left blank, never stealing a column an exact header match already
    claimed.

    Args:
        df: The uploaded file, unmodified.

    Returns:
        ``{canonical_name: source_column_or_None}`` for every required and
        optional column -- the same shape as :func:`suggest_column_mapping`.
    """
    mapping = suggest_column_mapping(list(df.columns))
    used = {source for source in mapping.values() if source}
    for field, guess in guess_column_mapping(df).items():
        if mapping.get(field) is None and guess is not None and guess not in used:
            mapping[field] = guess
            used.add(guess)
    return mapping


def preview_mapped_row(
    raw_df: pd.DataFrame, mapping: dict[str, str | None], index: int = 0
) -> str:
    """Render one example row as it would look under the current mapping.

    Lets the user see immediately whether the mapping is right -- e.g.
    ``"Company: NTPC | Code: NTPC-VLV-2201 | Category: Gate Valve |
    Description: GATE V/V 100 NB CL-150 CS BODY"`` -- instead of finding out
    after a full pipeline run.

    Args:
        raw_df: The uploaded file, unmodified.
        mapping: ``{canonical_name: source_column_or_None}``.
        index: Row to show; clamped into range.

    Returns:
        A single ``" | "``-joined line, or the empty string for an empty
        frame.
    """
    if raw_df.empty:
        return ""
    index = max(0, min(int(index), len(raw_df) - 1))
    row = raw_df.iloc[index]
    parts: list[str] = []
    for field, label in (
        ("CPSE", "Company"),
        ("CPSE Material Code", "Code"),
        ("Material Category", "Category"),
        (config.INPUT_TEXT_COLUMN, "Description"),
    ):
        source = mapping.get(field)
        if source and source in raw_df.columns:
            value = row[source]
            text = "" if value is None else str(value).strip()
            cell = text or "(blank)"
        else:
            cell = "(unmapped)"
        parts.append(f"{label}: {cell}")
    return " | ".join(parts)


def duplicate_required_sources(
    mapping: dict[str, str | None]
) -> dict[str, list[str]]:
    """Required fields that have been pointed at the same source column.

    The mapping UI offers every source column for every field, so nothing
    stops a user picking one column for two (or four) required fields. That
    passes :func:`apply_column_mapping`'s existence checks -- the column is
    real -- but produces a dataset where, say, ``Raw Description`` and
    ``Material Category`` are identical, and the pipeline then quietly finds
    no matches. This catches it before the run, not after.

    Args:
        mapping: ``{canonical_name: source_column_or_None}`` as confirmed by
            the user.

    Returns:
        ``{source_column: [required fields mapped to it]}`` for every source
        column claimed by two or more required fields, each field list in
        :data:`REQUIRED_COLUMNS` order. Empty when the required mapping is
        one-to-one.
    """
    by_source: dict[str, list[str]] = {}
    for field in REQUIRED_COLUMNS:
        source = mapping.get(field)
        if source:
            by_source.setdefault(source, []).append(field)
    return {
        source: fields for source, fields in by_source.items() if len(fields) > 1
    }


def apply_column_mapping(
    raw_df: pd.DataFrame, mapping: dict[str, str | None]
) -> pd.DataFrame:
    """Rename mapped columns onto the canonical schema, blanking the rest.

    Args:
        raw_df: The uploaded file, unmodified.
        mapping: ``{canonical_name: source_column_or_None}``, as confirmed by
            the user (see :func:`suggest_column_mapping`). Every entry in
            :data:`REQUIRED_COLUMNS` must map to a real column.

    Returns:
        A frame indexed 0..n-1 with exactly the canonical column names the
        rest of the pipeline expects: mapped required/optional columns as
        given, unmapped optional columns blank, plus the auto-defaulted
        columns (UOM, Legacy_Sector_Code) blank.

    Raises:
        ValueError: If a required column has no mapping, or if the same
            source column is mapped to more than one required field.
    """
    missing_required = [c for c in REQUIRED_COLUMNS if not mapping.get(c)]
    if missing_required:
        raise ValueError(
            f"Required column(s) not mapped: {', '.join(missing_required)}."
        )

    collisions = duplicate_required_sources(mapping)
    if collisions:
        detail = "; ".join(
            f"{source!r} -> {', '.join(fields)}"
            for source, fields in collisions.items()
        )
        raise ValueError(
            f"One source column is mapped to several required fields: {detail}."
        )

    out = pd.DataFrame(index=raw_df.index)
    for target in ALL_MAPPABLE_COLUMNS:
        source = mapping.get(target)
        out[target] = raw_df[source] if source else ""
    for column, default in _AUTO_DEFAULTED_COLUMNS.items():
        out[column] = default
    return out.reset_index(drop=True)


def build_dataset_from_mapped(
    mapped_df: pd.DataFrame, source_label: str
) -> ingestion.MaterialDataset:
    """Wrap a mapped upload as a MaterialDataset, matching load_dataset()'s contract.

    An upload never carries ``GroundTruth_Group`` -- ``has_labels`` is
    always False, and ``eval_df``/``sector_reference`` are empty rather than
    the demo workbook's evaluation columns and CPSE lookup sheet.

    Args:
        mapped_df: Output of :func:`apply_column_mapping`.
        source_label: Original filename, recorded as ``source_path`` for
            display only.

    Returns:
        A :class:`ingestion.MaterialDataset` ready for
        ``attribute_extraction.extract_frame`` and the rest of the pipeline.
    """
    dataset = ingestion.MaterialDataset(
        pipeline_df=mapped_df,
        eval_df=pd.DataFrame(index=mapped_df.index),
        sector_reference=pd.DataFrame(columns=["Sector", "CPSE"]),
        profile=ingestion.profile_dataframe(mapped_df),
        source_path=Path(source_label),
        has_labels=False,
    )
    dataset.assert_no_leakage()
    return dataset


def estimate_runtime_seconds(n_rows: int) -> float:
    """Estimate full-pipeline runtime for a given record count.

    A ROUGH ESTIMATE, not a guarantee: extrapolated from the one measured
    point (``config.RUNTIME_BASELINE_ROWS`` records in
    ``config.RUNTIME_BASELINE_SECONDS``, see README.md section 5) using a
    super-linear power law -- see ``config.RUNTIME_SCALING_EXPONENT`` for
    why comparisons, and therefore runtime, grow faster than linearly with
    record count.

    Runtime is modelled as a fixed setup overhead
    (``config.RUNTIME_FIXED_OVERHEAD_SECONDS`` -- encoder import, classifier
    load, blocking structures) plus a variable term that scales the
    remainder of the measured baseline. The fixed term keeps the estimate
    from reporting ``~0s`` for a tiny file, where setup is the whole cost,
    while leaving the estimate at the baseline row count exactly equal to
    ``config.RUNTIME_BASELINE_SECONDS``.

    Args:
        n_rows: Records in the (possibly sampled) dataset.

    Returns:
        Estimated seconds for the full pipeline, never below
        ``config.RUNTIME_FIXED_OVERHEAD_SECONDS``.
    """
    ratio = n_rows / config.RUNTIME_BASELINE_ROWS
    variable_baseline = (
        config.RUNTIME_BASELINE_SECONDS - config.RUNTIME_FIXED_OVERHEAD_SECONDS
    )
    variable = variable_baseline * (ratio**config.RUNTIME_SCALING_EXPONENT)
    return config.RUNTIME_FIXED_OVERHEAD_SECONDS + variable
