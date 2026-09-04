"""
Data ingestion and profiling (Spec section 4.1).

WHAT THIS FILE DOES
    Loads the CPSE material master workbook, splits it into a *pipeline* frame
    (what the matcher is allowed to see) and an *evaluation* frame (labels the
    matcher must never see), and computes profiling statistics for the dashboard.

    The split is enforced structurally, not by convention: `Standardized
    Description` and `GroundTruth_Group` are physically removed from the frame
    returned as pipeline input, so leakage cannot happen by accident.

INPUTS
    Path to `cpse_synthetic_dataset_with_cpse_mapping.xlsx` (sheet `All_Materials`).

OUTPUTS
    MaterialDataset  - dataclass holding .pipeline_df, .eval_df, .profile
    ProfileStats     - dataclass of row counts, null rates, distinct UOMs, etc.

KEY FUNCTIONS
    load_dataset(path)          -> MaterialDataset
    profile_dataframe(df)       -> ProfileStats
    ground_truth_pairs(eval_df) -> set[tuple[int, int]]
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ProfileStats:
    """Descriptive statistics about an ingested material master.

    Attributes:
        n_rows: Total material records loaded.
        n_cpses: Distinct CPSEs represented.
        n_categories: Distinct material categories.
        rows_per_cpse: Record count keyed by CPSE.
        rows_per_sector: Record count keyed by sector.
        null_rates: Fraction of nulls (0-1) keyed by attribute column.
        distinct_uoms: Sorted list of unit-of-measure values in use.
        duplicate_raw_descriptions: Count of exactly repeated raw descriptions.
        duplicate_material_codes: Count of repeated CPSE material codes; a
            non-zero value means the source ERP export is itself inconsistent.
    """

    n_rows: int
    n_cpses: int
    n_categories: int
    rows_per_cpse: dict[str, int]
    rows_per_sector: dict[str, int]
    null_rates: dict[str, float]
    distinct_uoms: list[str]
    duplicate_raw_descriptions: int
    duplicate_material_codes: int

    def as_records(self) -> list[dict[str, object]]:
        """Return null rates as a list of dicts, ready for a Streamlit table.

        Returns:
            One dict per attribute column with keys ``column``, ``null_rate``
            and ``null_count``-friendly ``pct`` string.
        """
        return [
            {"column": col, "null_rate": rate, "pct": f"{rate:.1%}"}
            for col, rate in sorted(
                self.null_rates.items(), key=lambda kv: -kv[1]
            )
        ]


@dataclass
class MaterialDataset:
    """An ingested material master, split into pipeline and evaluation halves.

    Attributes:
        pipeline_df: Columns the matcher may read (identity, context, raw text,
            semi-structured attributes). Indexed 0..n-1 -- this positional index
            is the record id used by every downstream module.
        eval_df: Evaluation-only columns (`Standardized Description`,
            `GroundTruth_Group`), sharing the same index as ``pipeline_df``.
            Real CPSE uploads carry neither column, so this may be empty.
        sector_reference: The `Sector_CPSE_Reference` lookup sheet.
        profile: Profiling statistics for the dashboard.
        source_path: Where the workbook was loaded from.
        has_labels: Whether ``GroundTruth_Group`` was present in the source
            workbook. False for real CPSE data, which has no answer key --
            callers must check this before using any ground-truth helper
            below (``ground_truth_pairs``, ``ground_truth_summary``,
            ``find_cross_cpse_examples``) or training the classifier.
    """

    pipeline_df: pd.DataFrame
    eval_df: pd.DataFrame
    sector_reference: pd.DataFrame
    profile: ProfileStats
    source_path: Path
    has_labels: bool = True
    _leak_guard: tuple[str, ...] = field(default=tuple(config.EVAL_COLUMNS))

    def assert_no_leakage(self) -> None:
        """Raise if an evaluation-only column has crept into the pipeline frame.

        Called at the end of :func:`load_dataset` and safe to call again after
        any transformation that merges frames.

        Raises:
            AssertionError: If a forbidden column is present in ``pipeline_df``.
        """
        leaked = [c for c in self._leak_guard if c in self.pipeline_df.columns]
        if leaked:
            raise AssertionError(
                f"Evaluation-only column(s) leaked into pipeline input: {leaked}. "
                "These are the answer key and must never be visible to the matcher."
            )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path | str | None = None) -> MaterialDataset:
    """Load the material master workbook and split off the evaluation labels.

    Args:
        path: Workbook location. Defaults to ``config.DATASET_PATH``.

    Returns:
        A :class:`MaterialDataset` whose ``pipeline_df`` contains only columns
        the matcher is permitted to see. ``GroundTruth_Group`` is optional --
        real CPSE exports do not have it, and its absence is reported via
        ``dataset.has_labels`` rather than raised as an error.

    Raises:
        FileNotFoundError: If the workbook is missing.
        AssertionError: If the leakage guard trips.
    """
    path = Path(path) if path is not None else config.DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the workbook in data/ or pass an "
            "explicit path."
        )

    raw = pd.read_excel(path, sheet_name=config.MATERIALS_SHEET)
    raw = raw.reset_index(drop=True)

    try:
        sector_reference = pd.read_excel(
            path, sheet_name=config.SECTOR_REFERENCE_SHEET
        )
    except ValueError:
        # Sheet is a convenience lookup, not a hard requirement.
        sector_reference = pd.DataFrame(columns=["Sector", "CPSE"])

    eval_columns = [c for c in config.EVAL_COLUMNS if c in raw.columns]
    eval_df = raw[eval_columns].copy()
    pipeline_df = raw.drop(columns=eval_columns).copy()

    dataset = MaterialDataset(
        pipeline_df=pipeline_df,
        eval_df=eval_df,
        sector_reference=sector_reference,
        profile=profile_dataframe(raw),
        source_path=path,
        has_labels="GroundTruth_Group" in eval_df.columns,
    )
    dataset.assert_no_leakage()
    return dataset


def profile_dataframe(df: pd.DataFrame) -> ProfileStats:
    """Compute descriptive statistics used by the dashboard and the report.

    Args:
        df: The full ingested frame, before the evaluation split.

    Returns:
        A populated :class:`ProfileStats`.
    """
    attribute_cols = [c for c in config.ATTRIBUTE_COLUMNS if c in df.columns]
    null_rates = {col: float(df[col].isna().mean()) for col in attribute_cols}

    return ProfileStats(
        n_rows=int(len(df)),
        n_cpses=int(df["CPSE"].nunique()),
        n_categories=int(df["Material Category"].nunique()),
        rows_per_cpse=df["CPSE"].value_counts().to_dict(),
        rows_per_sector=df["Sector"].value_counts().to_dict(),
        null_rates=null_rates,
        distinct_uoms=sorted(df["UOM"].dropna().unique().tolist()),
        duplicate_raw_descriptions=int(
            df[config.INPUT_TEXT_COLUMN].duplicated().sum()
        ),
        duplicate_material_codes=int(df["CPSE Material Code"].duplicated().sum()),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def ground_truth_pairs(eval_df: pd.DataFrame) -> set[tuple[int, int]]:
    """Enumerate every true duplicate pair implied by ``GroundTruth_Group``.

    Two records form a true pair when they share a group id. Pairs are returned
    as ordered ``(lower_index, higher_index)`` tuples so membership tests are
    direction-independent.

    Args:
        eval_df: The evaluation frame from :class:`MaterialDataset`.

    Returns:
        Set of index pairs. On the supplied dataset this is 2,629 pairs out of
        12,537,528 possible -- a 0.02% positive rate, which is why blocking and
        a precision-aware threshold matter more than raw accuracy.

    Raises:
        ValueError: If ``eval_df`` has no ``GroundTruth_Group`` column --
            check ``dataset.has_labels`` before calling this.
    """
    if "GroundTruth_Group" not in eval_df.columns:
        raise ValueError(
            "No GroundTruth_Group column: this dataset has no ground-truth "
            "labels. Check dataset.has_labels before calling evaluation "
            "helpers."
        )
    pairs: set[tuple[int, int]] = set()
    for _, idx in eval_df.groupby("GroundTruth_Group").groups.items():
        members = sorted(int(i) for i in idx)
        if len(members) < 2:
            continue
        pairs.update(itertools.combinations(members, 2))
    return pairs


def ground_truth_summary(
    eval_df: pd.DataFrame, pipeline_df: pd.DataFrame
) -> dict[str, int]:
    """Summarise the label set, including the within- vs cross-CPSE split.

    The cross-CPSE figure is the one the problem statement actually cares about:
    the same physical material coded differently by two different enterprises.

    Args:
        eval_df: Evaluation frame containing ``GroundTruth_Group``.
        pipeline_df: Pipeline frame containing ``CPSE`` (same index).

    Returns:
        Dict with total groups, multi-member groups, cross-CPSE groups, and the
        pair-level breakdown.

    Raises:
        ValueError: If ``eval_df`` has no ``GroundTruth_Group`` column --
            check ``dataset.has_labels`` before calling this.
    """
    if "GroundTruth_Group" not in eval_df.columns:
        raise ValueError(
            "No GroundTruth_Group column: this dataset has no ground-truth "
            "labels. Check dataset.has_labels before calling evaluation "
            "helpers."
        )
    cpse = pipeline_df["CPSE"]
    grouped = eval_df.groupby("GroundTruth_Group")
    sizes = grouped.size()
    multi_member = sizes[sizes > 1]

    cross_cpse_groups = 0
    for gid in multi_member.index:
        members = eval_df.index[eval_df["GroundTruth_Group"] == gid]
        if cpse.loc[members].nunique() > 1:
            cross_cpse_groups += 1

    pairs = ground_truth_pairs(eval_df)
    cross_pairs = sum(1 for a, b in pairs if cpse.iat[a] != cpse.iat[b])

    return {
        "total_groups": int(sizes.size),
        "multi_member_groups": int(multi_member.size),
        "cross_cpse_groups": cross_cpse_groups,
        "true_pairs": len(pairs),
        "cross_cpse_pairs": cross_pairs,
        "within_cpse_pairs": len(pairs) - cross_pairs,
        "possible_pairs": len(pipeline_df) * (len(pipeline_df) - 1) // 2,
    }


def find_cross_cpse_examples(
    dataset: MaterialDataset, limit: int = 5, require_distinct_text: bool = True
) -> list[pd.DataFrame]:
    """Pull real cross-CPSE duplicate groups for the app's "The Problem" view.

    Args:
        dataset: The ingested dataset.
        limit: Maximum number of example groups to return.
        require_distinct_text: When True, only return groups where every raw
            description is textually different -- these are the persuasive
            examples, since an exact string match needs no AI to find.

    Returns:
        A list of small frames, each one equivalence group, with the CPSE code,
        raw description and standardized description side by side.

    Raises:
        ValueError: If ``dataset.has_labels`` is False.
    """
    if not dataset.has_labels:
        raise ValueError(
            "No GroundTruth_Group column: this dataset has no ground-truth "
            "labels, so there are no known duplicate groups to pull examples "
            "from."
        )
    joined = dataset.pipeline_df.join(dataset.eval_df)
    examples: list[pd.DataFrame] = []

    for _, group in joined.groupby("GroundTruth_Group"):
        if len(group) < 2 or group["CPSE"].nunique() < 2:
            continue
        if require_distinct_text:
            normalized = group[config.INPUT_TEXT_COLUMN].str.upper().str.strip()
            if normalized.nunique() != len(group):
                continue
        examples.append(
            group[
                [
                    "CPSE",
                    "CPSE Material Code",
                    "Material Category",
                    config.INPUT_TEXT_COLUMN,
                    "Standardized Description",
                ]
            ]
        )
        if len(examples) >= limit:
            break
    return examples


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    ds = load_dataset()
    print(f"Loaded {ds.profile.n_rows} rows from {ds.source_path.name}")
    print(f"Pipeline columns : {list(ds.pipeline_df.columns)}")
    print(f"Evaluation columns: {list(ds.eval_df.columns)}")
    print(ground_truth_summary(ds.eval_df, ds.pipeline_df))
