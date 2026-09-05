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
    read_upload(file, filename)           -> pd.DataFrame
    suggest_column_mapping(columns)       -> dict[str, str | None]
    apply_column_mapping(raw_df, mapping) -> pd.DataFrame
    build_dataset_from_mapped(df, label)  -> MaterialDataset
    estimate_runtime_seconds(n_rows)      -> float
"""

from __future__ import annotations

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
        ValueError: If a required column has no mapping.
    """
    missing_required = [c for c in REQUIRED_COLUMNS if not mapping.get(c)]
    if missing_required:
        raise ValueError(
            f"Required column(s) not mapped: {', '.join(missing_required)}."
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

    Args:
        n_rows: Records in the (possibly sampled) dataset.

    Returns:
        Estimated seconds for the full pipeline.
    """
    ratio = n_rows / config.RUNTIME_BASELINE_ROWS
    return config.RUNTIME_BASELINE_SECONDS * (ratio**config.RUNTIME_SCALING_EXPONENT)
