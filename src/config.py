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
# Tier cut-offs are CALIBRATED, not hard-coded -- see
# classifier.calibrate_tiers. A fixed 0.85/0.65 pair is only valid on the score
# scale it was written for, and this system has three such scales in play (the
# hand-fused score, and classifier probabilities under each semantic backend).
# What is fixed instead is the business question: how precise must a merge be
# before it happens without a human?
HIGH_TIER_PRECISION_TARGET = 0.95

# Cluster-level precision required before a merge happens with no human in the
# loop. Deliberately stricter than the F1-optimal operating point: F1 treats a
# missed duplicate and a wrongly merged pair of national codes as equally bad,
# and they are not. A missed duplicate is found on the next run; a bad merge
# puts two different materials behind one code across every CPSE that adopts it.
AUTO_APPROVE_PRECISION_TARGET = 0.99

# Edge threshold and HIGH tier for the UNLABELLED path (a real CPSE upload
# with no GroundTruth_Group). There is nothing to calibrate fresh cut-offs
# from -- classifier.calibrate_tiers needs validation labels and
# matching_engine.calibrate_tiers_from_sweep needs a truth-pair sweep -- so
# this path loads the shipped models/classifier.pkl and reuses cut-offs on
# its match_probability measured on the demo dataset instead (see
# notebooks/evaluation.ipynb): edge threshold 0.55 (cluster-optimal), HIGH
# >= 0.85. The model's features (similarity scores, attribute-agreement
# flags) are dataset-agnostic, so match_probability itself transfers to new
# data without retraining -- but these two cut-offs were calibrated
# specifically on the demo dataset and are a starting point, not a
# guarantee, for a materially different catalogue.
PRETRAINED_EDGE_THRESHOLD = 0.55
PRETRAINED_HIGH_THRESHOLD = 0.85

# Degraded last-resort fallback, used ONLY when models/classifier.pkl is
# missing and there is therefore no trained match probability at all --
# routes on the hand-fused score instead. MEASURED on the demo dataset with
# labels stripped and scored against its (withheld) ground truth: at
# threshold 0.65, precision is 0.024 with 4,753 of 5,008 records merged into
# 467 clusters; the best F1 reachable across a 0.65-0.92 sweep is 0.264,
# against 0.897 for the trained-classifier path. The fused score does not
# separate classes well enough to survive transitive chaining in
# clustering -- this fallback exists so the pipeline still runs end to end,
# not because its output is trustworthy. Callers MUST warn the user plainly
# when this path is in use.
FUSED_HIGH_CONFIDENCE_THRESHOLD = 0.85
FUSED_MEDIUM_CONFIDENCE_THRESHOLD = 0.65

# A pair with fewer than this many jointly-known attributes cannot be routed to
# HIGH regardless of text similarity; it goes to UNKNOWN / "insufficient data".
# 4,650 of 5,008 rows have a null Operating Parameter and 4,342 a null
# Specification/Standard, so auto-approving on text alone would be false
# confidence at scale.
MIN_KNOWN_ATTRIBUTES_FOR_AUTO = 2

# A connected component stays whole only if its mean internal edge weight is at
# least this multiple of the MEDIUM tier threshold. Expressed as a ratio rather
# than an absolute score so it, too, survives a change of scoring scale.
# Guards against transitive chaining (A~B, B~C, A!~C) collapsing a category
# into one giant cluster.
MIN_CLUSTER_COHESION_RATIO = 1.0

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------
# Union of key families; see src/blocking.py for what each one covers.
# Benchmarked on the full 5,008-row dataset (src.blocking.compare_schemes):
#
#   scheme                    comparisons  reduction  recall ceiling  pairs lost
#   none (all pairs)           12,537,528       1.0x         100.00%           0
#   category_size only            174,466      71.9x          98.21%          47
#   category_token only           185,088      67.7x          97.72%          60
#   attribute_signature only       11,123    1127.2x           4.64%       2,507
#   category_size + token         193,924      64.7x         100.00%           0
#   union (all three)             194,400      64.5x         100.00%           0
#
# category_token costs 11% more comparisons than category_size alone and buys
# back all 47 pairs that exact-category blocking cannot reach. Worth it.
#
# attribute_signature is implemented and available, but on THIS dataset it adds
# 476 comparisons and zero pairs, because all five sectors draw category names
# from one shared vocabulary -- an artifact of the synthetic data. Enable it for
# a real multi-CPSE load where taxonomies are genuinely disjoint and the
# category keys have nothing to agree on:
#     BLOCKING_KEYS = ["category_size", "category_token", "attribute_signature"]
BLOCKING_KEYS = ["category_size", "category_token"]
SIZE_BUCKET_MM = 25.0    # coarse rounding for the category key
MAX_BLOCK_SIZE = 400     # a block larger than this is sub-split; prevents one
                         # huge category from dominating runtime
TOP_K_CANDIDATES = 20    # per-record candidate cap after scoring

# ---------------------------------------------------------------------------
# Upload runtime estimation ("Load Data" view)
# ---------------------------------------------------------------------------
# One measured point (README.md section 5): the full pipeline -- blocking
# through clustering -- on the bundled demo dataset. Comparisons, and
# therefore runtime, grow faster than linearly with record count: blocking
# buckets are keyed by category (src/blocking.py), and a bigger dataset means
# bigger buckets, not just more of them, so each bucket's internal
# comparisons grow with the square of its own size. RUNTIME_SCALING_EXPONENT
# extrapolates from that single measurement for the upload view's warning; it
# is an estimate to inform a sampling decision, not a benchmarked guarantee.
RUNTIME_BASELINE_ROWS = 5_008
RUNTIME_BASELINE_SECONDS = 80
RUNTIME_SCALING_EXPONENT = 1.3

# Above this many rows, the "Load Data" view warns and offers to sample --
# the super-linear growth above means a much bigger file is not just
# proportionally slower to demo.
LARGE_DATASET_WARNING_ROWS = 20_000

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
# Review-feedback tuning. The review queue is uncertainty-sampled, so
# reviewer-labelled pairs cluster around the decision boundary and are NOT a
# representative sample of the score distribution. Fitting a global threshold
# to them naively moved it 0.55 -> 0.19 and cost ~2 points of F1, so the
# recalibration is blended and gated on a minimum label count.
MIN_REVIEW_LABELS_FOR_RECALIBRATION = 100
REVIEW_RECALIBRATION_BLEND = 0.25

RANDOM_SEED = 42
# Ground-truth groups are split three ways, never pairs -- see
# classifier.group_disjoint_split. Validation exists so the decision threshold
# is tuned without ever touching the reported test set.
TEST_SPLIT_FRACTION = 0.30
VAL_SPLIT_FRACTION = 0.15
