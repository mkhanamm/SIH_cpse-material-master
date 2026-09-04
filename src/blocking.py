"""
Candidate blocking -- run before any expensive similarity (Spec section 4.4).

WHAT THIS FILE DOES
    Reduces the O(n^2) comparison space to a tractable candidate set by bucketing
    records on cheap keys, then emitting only within-bucket pairs.

    Three key families are used as a UNION, because no single one suffices:
      - CATEGORY key: exact `Material Category` + coarse size bucket. Cheap and
        precise, but it inherits every taxonomy disagreement between CPSEs.
        Measured cost on this dataset: 47 true pairs unreachable.
      - CATEGORY-TOKEN key: each meaningful token of the category name, so
        `Conveyor Idler` and `Conveyor Component` meet in the `conveyor` bucket
        and `Globe Valve` meets `Valve` in the `valve` bucket. This is what
        actually recovers the 47 pairs above -- they are naming inconsistencies,
        not attribute-only matches.
      - ATTRIBUTE-SIGNATURE key: a category-independent fingerprint (size band,
        spec family, material, capacity). Contributes little on this dataset,
        where categories are at least drawn from one shared vocabulary; it is
        retained because real CPSE masters use entirely disjoint taxonomies,
        which is precisely the case the first two keys cannot handle.

    A record may land in several buckets per family (multi-key blocking); pairs
    are deduplicated across all families before scoring.

    Reports honest metrics: comparisons before/after, reduction factor, and the
    *recall ceiling* the blocking scheme imposes on the ground-truth pairs --
    a speedup number without its recall cost is not a real measurement.

INPUTS
    DataFrame with `Material Category` and extracted attribute columns.

OUTPUTS
    BlockingResult - .pairs (candidate index pairs), .blocks, .stats
    BlockingStats  - naive_comparisons, blocked_comparisons, reduction_factor,
                     recall_ceiling, pairs_lost

KEY FUNCTIONS
    build_blocks(df)                       -> dict[str, list[int]]
    candidate_pairs(df)                    -> BlockingResult
    evaluate_blocking(result, truth_pairs) -> BlockingStats
"""

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class BlockingStats:
    """Cost and quality of a blocking scheme.

    Both halves are reported together on purpose. A reduction factor quoted
    without its recall ceiling is not a measurement -- any scheme can achieve
    an arbitrarily large speedup by discarding candidates.

    Attributes:
        n_records: Records blocked.
        naive_comparisons: n*(n-1)/2, the all-pairs baseline.
        blocked_comparisons: Distinct candidate pairs actually emitted.
        reduction_factor: naive / blocked.
        n_blocks: Number of non-singleton buckets produced.
        largest_block: Size of the biggest bucket after sub-splitting.
        true_pairs: Ground-truth duplicate pairs available.
        true_pairs_retained: How many survived blocking.
        recall_ceiling: retained / true_pairs -- the highest recall any
            downstream matcher can now achieve, however good it is.
        pairs_lost: true_pairs - true_pairs_retained.
    """

    n_records: int
    naive_comparisons: int
    blocked_comparisons: int
    reduction_factor: float
    n_blocks: int
    largest_block: int
    true_pairs: int = 0
    true_pairs_retained: int = 0
    recall_ceiling: float = 0.0
    pairs_lost: int = 0

    def summary_line(self) -> str:
        """Render the headline metric for the app and the README.

        Returns:
            A one-line human-readable summary.
        """
        return (
            f"{self.naive_comparisons:,} comparisons -> {self.blocked_comparisons:,} "
            f"after blocking ({self.reduction_factor:.1f}x reduction), "
            f"recall ceiling {self.recall_ceiling:.2%}"
        )


@dataclass
class BlockingResult:
    """Candidate pairs plus the bucket structure that produced them.

    Attributes:
        pairs: Sorted list of ``(lower_index, higher_index)`` candidate pairs,
            deduplicated across key families.
        blocks: Bucket key -> member indices, for inspection in the app.
        stats: Cost/quality metrics.
        pair_sources: Pair -> set of key families that proposed it. Used to show
            which blocking strategy earned each candidate.
    """

    pairs: list[tuple[int, int]]
    blocks: dict[str, list[int]]
    stats: BlockingStats
    pair_sources: dict[tuple[int, int], set[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------
def _size_bucket(value: object, width: float = config.SIZE_BUCKET_MM) -> str:
    """Round a numeric size into a coarse bucket label.

    Coarse on purpose: blocking must be recall-oriented, so a 40 mm and a 50 mm
    record should still land together and let the similarity stage separate
    them. Exact size agreement is decided later, with tolerance.

    Args:
        value: Extracted size in mm, or None.
        width: Bucket width in mm.

    Returns:
        A bucket label, or ``"na"`` when the size is unknown.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "na"
    try:
        return str(int(float(value) // width))
    except (TypeError, ValueError):
        return "na"


def _spec_family(value: object) -> str:
    """Reduce a specification string to its issuing-body family.

    ``astm a106 grade b`` and ``astm a106`` share the family ``astm-a106``, so
    they block together; ``is 2062`` does not.

    Args:
        value: Extracted spec/standard string, or None.

    Returns:
        A family label, or ``"na"``.
    """
    if not isinstance(value, str) or not value.strip():
        return "na"
    tokens = value.lower().replace("-", " ").split()
    return "-".join(tokens[:2]) if tokens else "na"


def category_key(row: pd.Series) -> str:
    """Build the exact-category blocking key for one record.

    Key = material category + coarse size bucket. Precise and cheap, but it can
    only match records whose CPSEs agreed on a category name, which is why the
    other two families exist.

    Args:
        row: A row carrying ``Material Category`` and ``nominal_size_mm``.

    Returns:
        The bucket key.
    """
    category = str(row.get("Material Category", "unknown")).strip().lower()
    return f"cat::{category}|sz::{_size_bucket(row.get('nominal_size_mm'))}"


# Head nouns too generic to block on. "Component" appears in eight different
# categories spanning ~900 records; bucketing on it would add a large number of
# comparisons between a boiler part and a conveyor part for no recall gain.
GENERIC_CATEGORY_TOKENS = frozenset(
    {
        "component", "components", "spare", "spares", "part", "parts",
        "equipment", "industrial", "type", "misc", "general", "assembly",
    }
)


def category_token_key(row: pd.Series) -> list[str]:
    """Build one key per meaningful token of the category name.

    This is the family that handles taxonomy disagreement between enterprises.
    Measured on this dataset, all 47 true pairs unreachable under exact-category
    blocking come from three near-synonym category pairs:
    `Conveyor Component` / `Conveyor Idler` (32 pairs), `Globe Valve` / `Valve`
    (12) and `Gate Valve` / `Valve` (3). Splitting the category name into tokens
    puts each of those pairs in a shared bucket -- `conveyor` and `valve`
    respectively -- without needing any attribute to be present.

    Args:
        row: A row carrying ``Material Category`` and ``nominal_size_mm``.

    Returns:
        Bucket keys, one per non-generic token. Empty when every token is
        generic, in which case the exact-category key still covers the record.
    """
    category = str(row.get("Material Category", "")).strip().lower()
    size = _size_bucket(row.get("nominal_size_mm"))
    tokens = [
        t for t in re.split(r"[^a-z0-9]+", category)
        if t and t not in GENERIC_CATEGORY_TOKENS
    ]
    return [f"tok::{t}|sz::{size}" for t in tokens]


def attribute_signature_key(row: pd.Series) -> str | None:
    """Build the category-independent attribute-signature key for one record.

    Ignores the category name entirely and fingerprints the record by what it
    physically is: size band, specification family, material of construction,
    capacity band, electrical rating band.

    Returns ``None`` unless the record has a known size *and* at least one other
    known attribute. An earlier, looser version of this guard admitted records
    whose signature was almost entirely ``na``, which piled thousands of
    unrelated attribute-poor records into one bucket -- a large cost for no
    recall. An under-specified record is better served by the category keys.

    On this dataset the family contributes no pairs the other two miss, because
    all five sectors draw category names from one shared vocabulary. It is kept
    because that assumption is an artifact of the synthetic data: real CPSE
    masters maintain independent taxonomies, where a fingerprint over physical
    attributes is the only key that can bridge them.

    Args:
        row: A row carrying extracted attribute columns.

    Returns:
        The bucket key, or None if the signature would be uninformative.
    """
    size = _size_bucket(row.get("nominal_size_mm"))
    if size == "na":
        return None

    spec = _spec_family(row.get("spec_standard"))
    material = row.get("material_of_construction")
    material = str(material).strip().lower() if isinstance(material, str) else "na"
    capacity = _size_bucket(row.get("capacity_cum"), width=1.0)
    rating = _size_bucket(row.get("rating_kva"), width=10.0)

    others = [spec, material, capacity, rating]
    if all(p == "na" for p in others):
        return None
    return "sig::" + "|".join([size, *others])


KEY_BUILDERS = {
    "category_size": category_key,
    "category_token": category_token_key,
    "attribute_signature": attribute_signature_key,
}


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------
def build_blocks(
    df: pd.DataFrame, keys: list[str] | None = None
) -> dict[str, dict[str, list[int]]]:
    """Bucket every record under each enabled key family.

    A key builder may place one record in several buckets of the same family.

    Args:
        df: Frame joining the pipeline columns with extracted attributes.
        keys: Key family names; defaults to ``config.BLOCKING_KEYS``.

    Returns:
        ``{key_family: {bucket_key: [record indices]}}``. Singleton buckets are
        retained here (they are dropped when pairs are emitted) so the app can
        show how many records had no candidate at all.
    """
    keys = keys or config.BLOCKING_KEYS
    blocks: dict[str, dict[str, list[int]]] = {k: defaultdict(list) for k in keys}

    for idx, row in df.iterrows():
        for family in keys:
            produced = KEY_BUILDERS[family](row)
            # A builder may return one key, several keys, or None. Several keys
            # means the record is deliberately placed in multiple buckets --
            # standard multi-key blocking, and the mechanism that lets a record
            # be found under more than one plausible name.
            if produced is None:
                continue
            for key in (produced,) if isinstance(produced, str) else produced:
                blocks[family][key].append(int(idx))

    return {family: dict(buckets) for family, buckets in blocks.items()}


def _split_oversized(
    members: list[int], max_size: int = config.MAX_BLOCK_SIZE
) -> list[list[int]]:
    """Sub-split a bucket that is too large to compare exhaustively.

    A single category such as `Seamless Pipe` (319 records) is fine, but an
    unlucky key can still concentrate records. Splitting caps worst-case cost;
    the cost is that a few in-bucket pairs are separated, which is charged
    honestly to the recall ceiling rather than hidden.

    Args:
        members: Record indices in one bucket.
        max_size: Maximum bucket size before splitting.

    Returns:
        A list of sub-buckets, each at most ``max_size`` long.
    """
    if len(members) <= max_size:
        return [members]
    ordered = sorted(members)
    return [ordered[i : i + max_size] for i in range(0, len(ordered), max_size)]


def candidate_pairs(
    df: pd.DataFrame, keys: list[str] | None = None
) -> BlockingResult:
    """Emit the deduplicated union of within-bucket pairs across key families.

    Union rather than intersection: the two families are designed to cover each
    other's blind spots, so a pair proposed by either one is a candidate.

    Args:
        df: Frame joining pipeline columns with extracted attributes.
        keys: Key family names; defaults to ``config.BLOCKING_KEYS``.

    Returns:
        A :class:`BlockingResult`. ``stats`` is populated with cost metrics;
        recall metrics require :func:`evaluate_blocking`.
    """
    keys = keys or config.BLOCKING_KEYS
    blocks = build_blocks(df, keys)

    pair_sources: dict[tuple[int, int], set[str]] = defaultdict(set)
    flat_blocks: dict[str, list[int]] = {}
    largest = 0

    for family, buckets in blocks.items():
        for key, members in buckets.items():
            for part_no, part in enumerate(_split_oversized(members)):
                if len(part) < 2:
                    continue
                label = f"{family}/{key}" + (f"#{part_no}" if part_no else "")
                flat_blocks[label] = part
                largest = max(largest, len(part))
                for a, b in itertools.combinations(sorted(part), 2):
                    pair_sources[(a, b)].add(family)

    pairs = sorted(pair_sources)
    n = len(df)
    naive = n * (n - 1) // 2

    stats = BlockingStats(
        n_records=n,
        naive_comparisons=naive,
        blocked_comparisons=len(pairs),
        reduction_factor=(naive / len(pairs)) if pairs else float("inf"),
        n_blocks=len(flat_blocks),
        largest_block=largest,
    )
    return BlockingResult(
        pairs=pairs, blocks=flat_blocks, stats=stats, pair_sources=dict(pair_sources)
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_blocking(
    result: BlockingResult, truth_pairs: set[tuple[int, int]]
) -> BlockingStats:
    """Charge the blocking scheme for the true pairs it discarded.

    Args:
        result: Output of :func:`candidate_pairs`.
        truth_pairs: Ground-truth duplicate pairs from
            ``ingestion.ground_truth_pairs``.

    Returns:
        The same stats object, with recall fields populated.
    """
    candidate_set = set(result.pairs)
    retained = len(truth_pairs & candidate_set)

    stats = result.stats
    stats.true_pairs = len(truth_pairs)
    stats.true_pairs_retained = retained
    stats.recall_ceiling = retained / len(truth_pairs) if truth_pairs else 0.0
    stats.pairs_lost = len(truth_pairs) - retained
    return stats


def compare_schemes(
    df: pd.DataFrame, truth_pairs: set[tuple[int, int]]
) -> pd.DataFrame:
    """Benchmark each key family alone against the union, for the report.

    This is the table that justifies building a second key family instead of
    asserting that one was needed.

    Args:
        df: Frame joining pipeline columns with extracted attributes.
        truth_pairs: Ground-truth duplicate pairs.

    Returns:
        A frame with one row per scheme: comparisons, reduction factor, recall
        ceiling and pairs lost.
    """
    schemes = {
        "none (all pairs)": None,
        "category_size only": ["category_size"],
        "category_token only": ["category_token"],
        "attribute_signature only": ["attribute_signature"],
        "category_size + token": ["category_size", "category_token"],
        "union (all three)": ["category_size", "category_token", "attribute_signature"],
    }

    rows = []
    for name, keys in schemes.items():
        if keys is None:
            n = len(df)
            naive = n * (n - 1) // 2
            rows.append(
                {
                    "scheme": name,
                    "comparisons": naive,
                    "reduction_factor": 1.0,
                    "recall_ceiling": 1.0,
                    "true_pairs_lost": 0,
                    "largest_block": n,
                }
            )
            continue
        result = candidate_pairs(df, keys)
        stats = evaluate_blocking(result, truth_pairs)
        rows.append(
            {
                "scheme": name,
                "comparisons": stats.blocked_comparisons,
                "reduction_factor": round(stats.reduction_factor, 1),
                "recall_ceiling": round(stats.recall_ceiling, 4),
                "true_pairs_lost": stats.pairs_lost,
                "largest_block": stats.largest_block,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from . import attribute_extraction, ingestion

    dataset = ingestion.load_dataset()
    extracted = attribute_extraction.extract_frame(dataset.pipeline_df)
    joined = dataset.pipeline_df.join(extracted)
    truth = ingestion.ground_truth_pairs(dataset.eval_df)

    print(compare_schemes(joined, truth).to_string(index=False))
