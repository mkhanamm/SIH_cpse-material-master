"""
Single source of every tunable constant.

WHAT THIS FILE DOES
    Centralises thresholds, weights, paths and backend selection so behaviour
    can be inspected and adjusted in one place instead of hunting across
    modules. Nothing here contains logic -- only named values and their
    justification comments.

SECTIONS
    Paths | Column names | Semantic backend | Similarity weights |
    Attribute tolerances | Confidence thresholds | Blocking parameters |
    CNMC schema | Impact-estimate assumptions
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"

DATASET_PATH = DATA_DIR / "cpse_synthetic_dataset_with_cpse_mapping.xlsx"
MATERIALS_SHEET = "All_Materials"
SECTOR_REFERENCE_SHEET = "Sector_CPSE_Reference"

CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
MAPPING_TABLE_PATH = OUTPUTS_DIR / "cnmc_mapping.csv"
AUDIT_LOG_PATH = OUTPUTS_DIR / "audit_log.jsonl"
REVIEW_LOG_PATH = OUTPUTS_DIR / "review_decisions.jsonl"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
# Identity columns: preserved end-to-end so every national code stays traceable
# back to the exact CPSE record it came from.
ID_COLUMNS = ["CPSE", "CPSE Material Code", "Legacy_Sector_Code"]

# Columns the matching pipeline is allowed to read.
INPUT_TEXT_COLUMN = "Raw Description"
CONTEXT_COLUMNS = ["Sector", "Material Category", "UOM"]
ATTRIBUTE_COLUMNS = [
    "Material/Grade",
    "Dimensions",
    "Specification/Standard",
    "Capacity/Rating",
    "Operating Parameter",
]

# Evaluation-only columns. These are split off at ingestion and never reach the
# pipeline frame -- `Standardized Description` is the answer key for the text
# task, and `GroundTruth_Group` is derived from it (verified 1:1 on this data:
# 3,965 groups == 3,965 distinct standardized descriptions). Feeding either one
# forward would leak the label.
EVAL_COLUMNS = ["Standardized Description", "GroundTruth_Group"]

# ---------------------------------------------------------------------------
# Semantic backend
# ---------------------------------------------------------------------------
# "sbert"     -> sentence-transformers; the production choice.
# "tfidf_svd" -> offline scikit-learn fallback, no model download needed.
# "auto"      -> try sbert, fall back to tfidf_svd if unavailable/offline.
SEMANTIC_BACKEND = "auto"
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_SVD_COMPONENTS = 256  # latent dimensions for the offline encoder

# ---------------------------------------------------------------------------
# Similarity channel weights
# ---------------------------------------------------------------------------
# Justification: the research finding is that embeddings win on paraphrase and
# word reordering, while string methods win on abbreviation and unit variants.
# Inspection of `Raw Description` in THIS dataset shows the duplicate signal is
# dominated by abbreviation variance ("BFLY V/V" vs "BUTTERFLY VALVE",
# "BRG" vs "BEARING", "SMLS" vs "SEAMLESS") rather than genuine paraphrase, so
# the string channel is weighted slightly higher. The attribute channel carries
# real weight because the hard negatives here are near-identical strings that
# differ in exactly one spec value (Schedule 20 vs Schedule 40).
CHANNEL_WEIGHTS = {
    "semantic": 0.35,
    "string": 0.40,
    "attribute": 0.25,
}

# A directly conflicting attribute (both sides known, values disagree) subtracts
# this much from the fused score. Without a veto term, a 0.95 text similarity
# would carry a Schedule-20/Schedule-40 mismatch over the threshold.
ATTRIBUTE_CONFLICT_PENALTY = 0.30

# ---------------------------------------------------------------------------
# Attribute comparison tolerances
# ---------------------------------------------------------------------------
DIMENSION_TOLERANCE_PCT = 0.02  # 40 mm vs 40.5 mm -> match; 40 vs 50 -> conflict
CAPACITY_TOLERANCE_PCT = 0.02
# Attributes that must never be treated as "close enough" -- a different
# pressure class or schedule is a different material for procurement purposes.
EXACT_MATCH_ATTRIBUTES = ["schedule", "pressure_class", "grade", "spec_standard"]

# ---------------------------------------------------------------------------
# Confidence thresholds (fused score, 0-1)
# ---------------------------------------------------------------------------
HIGH_CONFIDENCE_THRESHOLD = 0.85    # auto-suggest a CNMC
MEDIUM_CONFIDENCE_THRESHOLD = 0.65  # queue for human review
# Below MEDIUM -> no match asserted.

# A pair with fewer than this many jointly-known attributes cannot be routed to
# HIGH regardless of text similarity; it goes to UNKNOWN / "insufficient data".
# 4,650 of 5,008 rows have a null Operating Parameter and 4,342 a null
# Specification/Standard, so auto-approving on text alone would be false
# confidence at scale.
MIN_KNOWN_ATTRIBUTES_FOR_AUTO = 2

# Minimum average internal edge score for a connected component to stay whole.
# Guards against transitive chaining (A~B, B~C, A!~C) collapsing a category
# into one giant cluster.
MIN_CLUSTER_COHESION = 0.70

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------
# Union of two key families; see src/blocking.py for why one is not enough.
BLOCKING_KEYS = ["category_size", "attribute_signature"]
SIZE_BUCKET_MM = 25.0    # coarse rounding for the category key
MAX_BLOCK_SIZE = 400     # a block larger than this is sub-split; prevents one
                         # huge category from dominating runtime
TOP_K_CANDIDATES = 20    # per-record candidate cap after scoring

# ---------------------------------------------------------------------------
# CNMC schema
# ---------------------------------------------------------------------------
CNMC_PREFIX = "NM"
CNMC_SEQUENCE_WIDTH = 5
SECTOR_CODES = {
    "Oil & Gas": "OG",
    "Power": "PW",
    "Steel": "ST",
    "Mining": "MN",
    "Heavy Engineering": "HE",
}

# ---------------------------------------------------------------------------
# Impact estimate assumptions
# ---------------------------------------------------------------------------
# Every number the dashboard shows as "estimated savings" is derived from these
# three constants and is labelled as an estimate with the assumptions visible.
# They are illustrative planning figures, not audited procurement data.
AVG_LINE_ITEM_VALUE_INR = 250_000       # assumed average annual spend per code
DUPLICATE_CARRYING_COST_PCT = 0.08      # excess inventory carried per duplicate
CONSOLIDATION_PRICE_BENEFIT_PCT = 0.03  # volume leverage from consolidated demand

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SPLIT_FRACTION = 0.30
