"""
Trained duplicate classifier over similarity features (Spec section 4.5).

WHAT THIS FILE DOES
    Learns the pair -> duplicate decision instead of hand-tuning a threshold.
    For each candidate pair it builds a feature vector from the similarity
    channels (semantic, string, per-attribute agreement flags, missingness
    indicators), labels it from `GroundTruth_Group` (same group = duplicate),
    splits GROUP-WISE so that no ground-truth group appears in both train and
    test, fits a small model, and reports held-out precision/recall/F1.

    Group-wise splitting matters: a random pair-level split would put two pairs
    from the same cluster on both sides and inflate the score. The honest number
    is the one where the test clusters were never seen. Pairs that straddle the
    split (one record's group in train, the other's in test) are discarded
    rather than assigned arbitrarily -- keeping them would leak test clusters
    into training through the negative examples.

    Model is deliberately small and inspectable -- logistic regression by default
    (coefficients readable as evidence weights), gradient-boosted trees optional.

INPUTS
    Scored candidate pairs + evaluation labels.

OUTPUTS
    models/classifier.pkl  - fitted sklearn pipeline
    ClassifierReport       - precision/recall/F1 overall and split by
                             within-CPSE vs cross-CPSE, plus feature weights.

KEY FUNCTIONS
    build_features(scored_pairs)          -> (X, feature_names)
    label_pairs(pairs, eval_df)           -> ndarray
    train(X, y, groups)                   -> (model, ClassifierReport)
    save_model(model) / load_model()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# Feature columns fed to the model. Deliberately few and each individually
# meaningful, so a logistic-regression coefficient can be read as "how much this
# kind of evidence counts" and shown to a reviewer.
FEATURE_COLUMNS = [
    "semantic",            # embedding cosine
    "string",              # Jaro-Winkler + token-set blend
    "attribute",           # attribute agreement ratio
    "n_comparable_attrs",  # how much attribute evidence existed at all
    "n_agree_attrs",
    "n_conflicts",
    "hard_conflict",       # conflict on a procurement-critical attribute
    "min_known_attrs",     # documentation depth of the poorer record
    "same_category",
]


@dataclass
class ClassifierReport:
    """Held-out performance of a trained duplicate classifier.

    Attributes:
        backend: Semantic backend that produced the features. Metrics are not
            comparable across backends, so this always travels with them.
        model_name: Estimator used.
        n_train_pairs / n_val_pairs / n_test_pairs: Pair counts after
            group-disjoint splitting.
        n_discarded_pairs: Pairs straddling a split boundary, dropped to
            prevent leakage.
        validation_f1: Best F1 reached while tuning the threshold on validation.
            Compare against ``f1`` to see how well the tuned operating point
            generalised to unseen clusters.
        precision / recall / f1: On the held-out test pairs.
        average_precision: Area under the precision-recall curve -- the honest
            summary for a 1.4%-positive problem, where accuracy is meaningless.
        cross_cpse / within_cpse: The same three metrics restricted to each
            subset. Cross-CPSE is the case the problem statement cares about.
        feature_weights: Coefficient (or importance) per feature.
        threshold: Decision threshold used for the reported metrics.
        tiers: Calibrated HIGH/MEDIUM cut-offs for confidence routing.
    """

    backend: str
    model_name: str
    n_train_pairs: int
    n_val_pairs: int
    n_test_pairs: int
    n_discarded_pairs: int
    precision: float
    recall: float
    f1: float
    average_precision: float
    cross_cpse: dict[str, float] = field(default_factory=dict)
    within_cpse: dict[str, float] = field(default_factory=dict)
    feature_weights: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    validation_f1: float = float("nan")
    tiers: "TierThresholds | None" = None

    def summary_lines(self) -> list[str]:
        """Render the report for the README, the app and the notebook.

        Returns:
            Human-readable lines.
        """
        return [
            f"Model: {self.model_name} (semantic backend: {self.backend})",
            f"Held-out pairs: {self.n_test_pairs:,} "
            f"(train {self.n_train_pairs:,}, val {self.n_val_pairs:,}, "
            f"discarded at split boundary {self.n_discarded_pairs:,})",
            f"Threshold {self.threshold:.2f} (tuned on validation, "
            f"val F1 {self.validation_f1:.3f})",
            f"Precision {self.precision:.3f} | Recall {self.recall:.3f} | "
            f"F1 {self.f1:.3f} | AP {self.average_precision:.3f}",
            f"  cross-CPSE : P {self.cross_cpse.get('precision', 0):.3f} "
            f"R {self.cross_cpse.get('recall', 0):.3f} "
            f"F1 {self.cross_cpse.get('f1', 0):.3f} "
            f"(n={int(self.cross_cpse.get('n_true', 0))} true pairs)",
            f"  within-CPSE: P {self.within_cpse.get('precision', 0):.3f} "
            f"R {self.within_cpse.get('recall', 0):.3f} "
            f"F1 {self.within_cpse.get('f1', 0):.3f} "
            f"(n={int(self.within_cpse.get('n_true', 0))} true pairs)",
            *(self.tiers.summary_lines() if self.tiers else []),
        ]


# ---------------------------------------------------------------------------
# Features and labels
# ---------------------------------------------------------------------------
def build_features(scored_pairs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Assemble the feature matrix from scored candidate pairs.

    Note that ``fused`` is deliberately excluded. It is a hand-weighted
    combination of the other features, so including it would let the model
    inherit the hand-tuned weights rather than learn its own -- and the point of
    training a classifier is to find out whether the hand weights were right.

    Args:
        scored_pairs: Output of ``similarity.score_pairs``.

    Returns:
        ``(X, feature_names)`` with X of shape ``(n_pairs, n_features)``.
    """
    frame = scored_pairs[FEATURE_COLUMNS].astype(float)
    return frame.to_numpy(), list(FEATURE_COLUMNS)


def label_pairs(
    scored_pairs: pd.DataFrame,
    eval_df: pd.DataFrame,
    reviewer_label_column: str | None = None,
) -> np.ndarray:
    """Label each candidate pair as duplicate or not.

    Labels come from ``GroundTruth_Group`` unless a reviewer has ruled on the
    pair, in which case the reviewer's verdict wins. In production there is no
    ground-truth column at all and the reviewer is the only label source, so the
    override is the primary path rather than an exception to it.

    Args:
        scored_pairs: Output of ``similarity.score_pairs``.
        eval_df: Evaluation frame carrying ``GroundTruth_Group``.
        reviewer_label_column: Optional column holding reviewer labels, where
            ``-1`` means "not reviewed".

    Returns:
        Integer array of 0/1 labels aligned with ``scored_pairs``.
    """
    groups = eval_df["GroundTruth_Group"]
    left = groups.reindex(scored_pairs["idx_a"]).to_numpy()
    right = groups.reindex(scored_pairs["idx_b"]).to_numpy()
    labels = (left == right).astype(int)

    if reviewer_label_column and reviewer_label_column in scored_pairs.columns:
        reviewed = scored_pairs[reviewer_label_column].to_numpy()
        has_verdict = reviewed >= 0
        labels = np.where(has_verdict, reviewed, labels).astype(int)
    return labels


def group_disjoint_split(
    scored_pairs: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_fraction: float = config.TEST_SPLIT_FRACTION,
    val_fraction: float = config.VAL_SPLIT_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split pairs three ways so no ground-truth group appears on two sides.

    Ground-truth groups -- not pairs -- are randomly assigned to train,
    validation or test. A pair joins a side only when *both* of its records'
    groups are on that side; pairs straddling any boundary are discarded.

    Without this, a cluster of five records contributes ten pairs; a random pair
    split would train on some and test on others, and the reported score would
    partly measure memorisation of clusters the model had already seen.

    The validation split exists so the decision threshold can be tuned without
    touching test. Tuning a threshold on the test set and then reporting test
    performance is the same leak in a different coat.

    Args:
        scored_pairs: Output of ``similarity.score_pairs``.
        eval_df: Evaluation frame carrying ``GroundTruth_Group``.
        test_fraction: Share of groups held out for reporting.
        val_fraction: Share of groups held out for threshold tuning.
        seed: RNG seed.

    Returns:
        ``(train_mask, val_mask, test_mask, n_discarded)``.
    """
    rng = np.random.default_rng(seed)
    unique_groups = eval_df["GroundTruth_Group"].unique()
    shuffled = rng.permutation(unique_groups)

    n_test = int(len(shuffled) * test_fraction)
    n_val = int(len(shuffled) * val_fraction)
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test : n_test + n_val])

    groups = eval_df["GroundTruth_Group"]
    left = groups.reindex(scored_pairs["idx_a"]).to_numpy()
    right = groups.reindex(scored_pairs["idx_b"]).to_numpy()

    def _side(members: set[str]) -> np.ndarray:
        listed = list(members)
        return np.isin(left, listed) & np.isin(right, listed)

    test_mask = _side(test_groups)
    val_mask = _side(val_groups)
    train_mask = ~np.isin(left, list(test_groups | val_groups)) & ~np.isin(
        right, list(test_groups | val_groups)
    )
    n_discarded = int(
        len(scored_pairs) - test_mask.sum() - val_mask.sum() - train_mask.sum()
    )
    return train_mask, val_mask, test_mask, n_discarded


def tune_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Pick the probability cut-off that maximises F1 on validation data.

    A fixed 0.5 cut-off is meaningless for a model fitted with balanced class
    weights on a 1.4%-positive problem: the weighting deliberately moves the
    operating point, and 0.5 no longer sits anywhere useful on the curve.

    Args:
        y_true: Validation labels.
        probabilities: Predicted duplicate probabilities.

    Returns:
        ``(best_threshold, best_f1)``. Falls back to ``(0.5, 0.0)`` when the
        validation split contains no positives.
    """
    if y_true.sum() == 0:
        return 0.5, 0.0

    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold, best_f1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Compute precision/recall/F1 on a subset of the test pairs.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        mask: Boolean subset selector.

    Returns:
        Dict with ``precision``, ``recall``, ``f1`` and ``n_true``.
    """
    if mask.sum() == 0 or y_true[mask].sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_true": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true[mask], y_pred[mask], average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_true": float(y_true[mask].sum()),
    }


@dataclass
class TierThresholds:
    """Score cut-offs separating the confidence tiers.

    These are *derived*, not chosen. A hand-picked pair of numbers like
    0.85/0.65 is only meaningful on the score scale it was written for: the
    hand-fused score and a classifier probability occupy completely different
    ranges (this model's F1-optimal cut-off is ~0.06), and the SBERT and
    tfidf_svd backends shift the distribution again. Deriving both thresholds
    from a precision target on validation data makes the tiers mean the same
    thing under any backend or scoring change.

    Attributes:
        high: Lowest score at which validation precision meets the auto-approve
            target. Above this, a match is proposed without a reviewer.
        medium: F1-optimal cut-off. Between medium and high, a human decides.
        high_precision_target: The target that produced ``high``.
        achieved_high_precision: Validation precision actually attained.
        achieved_high_recall: Validation recall at the high cut-off -- how much
            of the duplicate population can be cleared without human effort.
        calibrated: False when ``high``/``medium`` are the fixed fallback
            cut-offs (:func:`fallback_tiers`) rather than derived from a
            precision target on validation data -- i.e. there was no
            ground truth to calibrate against.
    """

    high: float
    medium: float
    high_precision_target: float
    achieved_high_precision: float
    achieved_high_recall: float
    calibrated: bool = True

    def summary_lines(self) -> list[str]:
        """Render the calibration for the app and the report.

        Returns:
            Human-readable lines.
        """
        if not self.calibrated:
            return [
                f"HIGH   >= {self.high:.3f}  (fixed fallback threshold -- no "
                "ground-truth labels to calibrate against) -> auto-suggest CNMC",
                f"MEDIUM >= {self.medium:.3f}  (fixed fallback threshold) -> "
                "human review",
                f"LOW    <  {self.medium:.3f}  -> no match asserted",
            ]
        return [
            f"HIGH   >= {self.high:.3f}  "
            f"(validation precision {self.achieved_high_precision:.3f} "
            f"vs target {self.high_precision_target:.2f}, "
            f"recall {self.achieved_high_recall:.3f}) -> auto-suggest CNMC",
            f"MEDIUM >= {self.medium:.3f}  (F1-optimal) -> human review",
            f"LOW    <  {self.medium:.3f}  -> no match asserted",
        ]


def calibrate_tiers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    high_precision_target: float = config.HIGH_TIER_PRECISION_TARGET,
) -> TierThresholds:
    """Derive tier cut-offs from a precision target on validation data.

    The auto-approve tier is defined by the question that actually matters to a
    CPSE: at what score can we merge two material codes without a human looking,
    and be wrong less than N% of the time? That is a precision target, so the
    threshold is read off the precision-recall curve rather than guessed.

    Args:
        y_true: Validation labels.
        probabilities: Predicted duplicate probabilities on validation.
        high_precision_target: Precision required for auto-approval.

    Returns:
        A populated :class:`TierThresholds`. If no cut-off reaches the target,
        ``high`` is set to the most precise available point and the achieved
        precision is reported honestly rather than the target being relaxed
        silently.
    """
    medium, _ = tune_threshold(y_true, probabilities)

    best = None
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        if predictions.sum() == 0:
            continue
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )
        if precision >= high_precision_target:
            best = (float(threshold), float(precision), float(recall))
            break
        if best is None or precision > best[1]:
            best = (float(threshold), float(precision), float(recall))

    high, achieved_precision, achieved_recall = best or (0.99, 0.0, 0.0)
    return TierThresholds(
        high=max(high, medium),
        medium=medium,
        high_precision_target=high_precision_target,
        achieved_high_precision=achieved_precision,
        achieved_high_recall=achieved_recall,
    )


def pretrained_tiers(
    high: float = config.PRETRAINED_HIGH_THRESHOLD,
    medium: float = config.PRETRAINED_EDGE_THRESHOLD,
) -> TierThresholds:
    """Tier cut-offs for the shipped pre-trained classifier on unlabelled data.

    The PRIMARY path for a dataset with no ``GroundTruth_Group``: neither
    :func:`calibrate_tiers` (needs validation labels) nor
    ``matching_engine.calibrate_tiers_from_sweep`` (needs a truth-pair sweep)
    can run without ground truth, so there is nothing to calibrate fresh
    cut-offs from. Loading ``models/classifier.pkl`` instead gets a
    ``match_probability`` that transfers to new data without retraining --
    its features (similarity scores, attribute-agreement flags) describe the
    pair, not the dataset -- but ``high``/``medium`` were themselves
    calibrated on the demo dataset (:data:`config.PRETRAINED_HIGH_THRESHOLD`,
    :data:`config.PRETRAINED_EDGE_THRESHOLD`) and are a starting point, not a
    guarantee, on a materially different catalogue.

    Args:
        high: Auto-approve cut-off on match_probability.
        medium: Human-review cut-off; also used as the graph edge threshold.

    Returns:
        A :class:`TierThresholds` with ``calibrated=False``.
    """
    return TierThresholds(
        high=high,
        medium=medium,
        high_precision_target=float("nan"),
        achieved_high_precision=float("nan"),
        achieved_high_recall=float("nan"),
        calibrated=False,
    )


def fallback_tiers(
    high: float = config.FUSED_HIGH_CONFIDENCE_THRESHOLD,
    medium: float = config.FUSED_MEDIUM_CONFIDENCE_THRESHOLD,
) -> TierThresholds:
    """Fixed tier cut-offs for the hand-fused score -- a DEGRADED last resort.

    Used only when a dataset has no ``GroundTruth_Group`` AND
    ``models/classifier.pkl`` is missing, so there is no trained match
    probability available at all; see :func:`pretrained_tiers` for the
    primary unlabelled path. MEASURED on the demo dataset with labels
    stripped and scored against its (withheld) ground truth: at threshold
    0.65, precision is 0.024 with 4,753 of 5,008 records merged into 467
    clusters; the best F1 reachable across a 0.65-0.92 sweep is 0.264,
    against 0.899 for the trained-classifier path. The fused score does not
    separate classes well enough to survive transitive chaining in
    clustering. Callers must warn the user plainly when this path is in use.

    Args:
        high: Auto-approve cut-off on the fused score.
        medium: Human-review cut-off on the fused score.

    Returns:
        A :class:`TierThresholds` with ``calibrated=False``.
    """
    return TierThresholds(
        high=high,
        medium=medium,
        high_precision_target=float("nan"),
        achieved_high_precision=float("nan"),
        achieved_high_recall=float("nan"),
        calibrated=False,
    )


def train(
    scored_pairs: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_name: str = "gradient_boosting",
    threshold: float | None = None,
    reviewer_label_column: str | None = None,
) -> tuple[Pipeline, ClassifierReport]:
    """Fit and evaluate a duplicate classifier on the scored candidate pairs.

    Class weighting is balanced because positives are ~1.4% of candidates after
    blocking; an unweighted fit would score well by predicting "not duplicate"
    almost always. The consequence is that 0.5 is not a meaningful cut-off, so
    the threshold is tuned on the validation split unless one is supplied.

    ``gradient_boosting`` is the default. On this data the linear model reaches
    an average precision of ~0.51 against ~0.91 for the trees, because the
    decision is not linear in these features -- "high string similarity AND zero
    attribute conflicts" is a conjunction a single hyperplane cannot express.
    ``logistic_regression`` is retained as the interpretable baseline, since its
    coefficients are directly readable as evidence weights.

    Args:
        scored_pairs: Output of ``similarity.score_pairs``.
        eval_df: Evaluation frame carrying ``GroundTruth_Group``.
        model_name: ``"logistic_regression"`` or ``"gradient_boosting"``.
        threshold: Fixed probability cut-off. When None, tuned on validation.
        reviewer_label_column: Column carrying human verdicts that override the
            ground-truth label; see :func:`label_pairs`.

    Returns:
        ``(fitted_pipeline, report)``.

    Raises:
        ValueError: On an unknown ``model_name``.
    """
    X, feature_names = build_features(scored_pairs)
    y = label_pairs(scored_pairs, eval_df, reviewer_label_column)
    train_mask, val_mask, test_mask, n_discarded = group_disjoint_split(
        scored_pairs, eval_df
    )
    # Evaluation always uses ground truth, never reviewer verdicts: scoring the
    # model against the labels it was just handed would guarantee improvement
    # and measure nothing.
    y_eval = label_pairs(scored_pairs, eval_df)

    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
    elif model_name == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        # No class_weight here. Threshold tuning already places the operating
        # point, so re-weighting is redundant and measurably slightly worse:
        # selecting on VALIDATION F1 (not test), unweighted scores 0.882 against
        # 0.876 balanced. The linear model keeps its weighting because it has no
        # tuning headroom to spare.
        estimator = HistGradientBoostingClassifier(
            max_iter=200,
            random_state=config.RANDOM_SEED,
        )
    else:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected 'logistic_regression' "
            "or 'gradient_boosting'."
        )

    pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    pipeline.fit(X[train_mask], y[train_mask])

    val_probabilities = pipeline.predict_proba(X[val_mask])[:, 1]
    tiers = calibrate_tiers(y[val_mask], val_probabilities)
    if threshold is None:
        threshold, tuned_f1 = tiers.medium, tiers.medium and tune_threshold(
            y[val_mask], val_probabilities
        )[1]
    else:
        tuned_f1 = float("nan")

    probabilities = pipeline.predict_proba(X[test_mask])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    y_test = y_eval[test_mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    test_pairs = scored_pairs[test_mask]
    cross_mask = test_pairs["cross_cpse"].to_numpy(dtype=bool)

    if model_name == "logistic_regression":
        weights = dict(
            zip(feature_names, pipeline.named_steps["model"].coef_[0].round(4))
        )
    else:
        weights = {}

    report = ClassifierReport(
        backend=str(scored_pairs.attrs.get("semantic_backend", "unknown")),
        model_name=model_name,
        n_train_pairs=int(train_mask.sum()),
        n_val_pairs=int(val_mask.sum()),
        n_test_pairs=int(test_mask.sum()),
        n_discarded_pairs=n_discarded,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        average_precision=float(average_precision_score(y_test, probabilities)),
        cross_cpse=_subset_metrics(y_test, predictions, cross_mask),
        within_cpse=_subset_metrics(y_test, predictions, ~cross_mask),
        feature_weights={k: float(v) for k, v in weights.items()},
        threshold=float(threshold),
        validation_f1=float(tuned_f1),
        tiers=tiers,
    )
    return pipeline, report


def predict_proba(pipeline: Pipeline, scored_pairs: pd.DataFrame) -> np.ndarray:
    """Score candidate pairs with a fitted classifier.

    Args:
        pipeline: Fitted pipeline from :func:`train` or :func:`load_model`.
        scored_pairs: Output of ``similarity.score_pairs``.

    Returns:
        Duplicate probability per pair.
    """
    X, _ = build_features(scored_pairs)
    return pipeline.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(pipeline: Pipeline, path=None) -> None:
    """Persist a fitted pipeline.

    Args:
        pipeline: Fitted pipeline.
        path: Destination; defaults to ``config.CLASSIFIER_PATH``.
    """
    path = path or config.CLASSIFIER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path=None) -> Pipeline:
    """Load a persisted pipeline.

    Args:
        path: Source; defaults to ``config.CLASSIFIER_PATH``.

    Returns:
        The fitted pipeline.

    Raises:
        FileNotFoundError: If no model artifact exists.
    """
    path = path or config.CLASSIFIER_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No classifier at {path}. Run `python -m src.classifier` to train one."
        )
    return joblib.load(path)


def load_model_or_none(path=None) -> Pipeline | None:
    """Load the shipped pre-trained classifier, or None if it isn't there.

    Used by the unlabelled pipeline path to decide between the pre-trained
    ``match_probability`` (:func:`pretrained_tiers`) and the degraded
    hand-fused fallback (:func:`fallback_tiers`) when there is no ground
    truth to train a fresh model from.

    Args:
        path: Source; defaults to ``config.CLASSIFIER_PATH``.

    Returns:
        The fitted pipeline, or None if no artifact exists.
    """
    try:
        return load_model(path)
    except FileNotFoundError:
        return None


if __name__ == "__main__":  # pragma: no cover - trains and saves the artifact
    from . import attribute_extraction, blocking, ingestion, similarity
    from .normalization import normalize_series

    dataset = ingestion.load_dataset()
    extracted = attribute_extraction.extract_frame(dataset.pipeline_df)
    joined = dataset.pipeline_df.join(extracted)
    text = normalize_series(joined[config.INPUT_TEXT_COLUMN])

    candidates = blocking.candidate_pairs(joined)
    scored = similarity.score_pairs(joined, candidates.pairs, text)

    for name in ("gradient_boosting", "logistic_regression"):
        model, report = train(scored, dataset.eval_df, model_name=name)
        print("\n".join(report.summary_lines()))
        if name == "logistic_regression":
            print("feature weights:", report.feature_weights)
        else:
            save_model(model)
        print()
