"""
Human review queue and active-learning feedback loop (Spec section 4.8).

WHAT THIS FILE DOES
    Manages the medium-confidence queue and closes the loop so the system
    measurably improves as it is used.

    Review model: each queued item carries the candidate cluster, its
    explanation, and a decision record (reviewer id, timestamp, action in
    {approve, reject, edit}, free-text comment).

    Active learning: accumulated decisions are fed back two ways --
      1. THRESHOLD RECALIBRATION - refit the decision threshold on the growing
         set of reviewer-labelled pairs to maximise F1.
      2. CANDIDATE RE-RANKING - reviewed pairs are added as training rows and
         the classifier is refit, changing the ordering of future candidates.
    `measure_improvement()` reports precision/recall before vs. after N reviewed
    decisions, so "the system gets smarter" is a number on screen, not a claim.

    Uncertainty sampling decides what to show first: pairs nearest the current
    decision boundary are the most informative to label, so the queue is sorted
    by |score - threshold| ascending rather than by score descending.

INPUTS
    Medium-confidence clusters from matching_engine; reviewer actions from app.py.

OUTPUTS
    outputs/review_decisions.jsonl - append-only decision log
    ActiveLearningReport           - before/after metrics

KEY FUNCTIONS
    build_queue(clusters)                     -> list[ReviewItem]
    record_decision(item_id, action, reviewer, comment) -> DecisionRecord
    recalibrate_threshold(decisions)          -> float
    measure_improvement(before, after)        -> ActiveLearningReport
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

APPROVE, REJECT, EDIT = "approve", "reject", "edit"
VALID_ACTIONS = (APPROVE, REJECT, EDIT)


@dataclass
class ReviewItem:
    """One cluster awaiting a human decision.

    Attributes:
        item_id: Stable id within a review session.
        cluster_id: The cluster being reviewed.
        members: Record indices proposed as one material.
        tier: Confidence tier that put it in the queue.
        score: Cluster score (weakest internal link).
        uncertainty: Distance from the decision boundary. Small means the model
            is least sure, and therefore that a human label here teaches it the
            most.
        cpses: CPSEs represented.
        explanation: Serialised :class:`explanation.Explanation`.
        reason: Why it was queued.
    """

    item_id: int
    cluster_id: int
    members: list[int]
    tier: str
    score: float
    uncertainty: float
    cpses: list[str] = field(default_factory=list)
    explanation: dict[str, object] = field(default_factory=dict)
    reason: str = ""


@dataclass
class DecisionRecord:
    """An immutable record of one reviewer decision.

    Attributes:
        item_id: Which queue item was decided.
        cluster_id: Which cluster.
        action: approve, reject or edit.
        reviewer: Identity of the person deciding. Required -- an unattributed
            decision cannot be audited.
        timestamp: ISO-8601 UTC.
        comment: Free-text justification.
        members: Records the decision covers. For ``edit`` this is the corrected
            membership, which is the whole point of the edit action: a reviewer
            who says "these four are one material but the fifth is not" supplies
            more information than a bare reject.
        original_members: Membership as proposed, retained so an edit records
            what was changed rather than only the outcome.
        score: Model score at decision time, kept for recalibration.
    """

    item_id: int
    cluster_id: int
    action: str
    reviewer: str
    timestamp: str
    comment: str = ""
    members: list[int] = field(default_factory=list)
    original_members: list[int] = field(default_factory=list)
    score: float = 0.0

    def labelled_pairs(self) -> list[tuple[int, int, int]]:
        """Convert the decision into supervised pair labels.

        This is what makes review worth more than a yes/no: an approval labels
        every internal pair positive; a rejection labels them negative; an edit
        labels pairs inside the corrected set positive and pairs joining a
        removed record to a kept one negative.

        Returns:
            ``(idx_a, idx_b, label)`` triples with ``idx_a < idx_b``.
        """
        import itertools

        triples: list[tuple[int, int, int]] = []
        if self.action == APPROVE:
            for a, b in itertools.combinations(sorted(self.members), 2):
                triples.append((a, b, 1))
        elif self.action == REJECT:
            for a, b in itertools.combinations(sorted(self.original_members), 2):
                triples.append((a, b, 0))
        elif self.action == EDIT:
            kept = set(self.members)
            removed = set(self.original_members) - kept
            for a, b in itertools.combinations(sorted(kept), 2):
                triples.append((a, b, 1))
            for a in sorted(kept):
                for b in sorted(removed):
                    triples.append((min(a, b), max(a, b), 0))
        return triples


@dataclass
class ActiveLearningReport:
    """Before/after effect of feeding reviewer decisions back into the model.

    Attributes:
        n_decisions: Reviewer decisions applied.
        n_labelled_pairs: Supervised pairs those decisions produced.
        threshold_before / threshold_after: Decision threshold either side.
        precision_before / precision_after: And recall and F1, on the same
            held-out evaluation set both times -- otherwise "improvement" is
            just a change of yardstick.
        recall_before / recall_after: As above.
        f1_before / f1_after: As above.
    """

    n_decisions: int
    n_labelled_pairs: int
    threshold_before: float
    threshold_after: float
    precision_before: float
    precision_after: float
    recall_before: float
    recall_after: float
    f1_before: float
    f1_after: float

    def delta_f1(self) -> float:
        """Change in F1 attributable to the reviewed decisions.

        Returns:
            ``f1_after - f1_before``. May be negative; a loop that can only
            report improvement is not measuring anything.
        """
        return self.f1_after - self.f1_before

    def summary_lines(self) -> list[str]:
        """Render the before/after comparison.

        Returns:
            Human-readable lines.
        """
        return [
            f"{self.n_decisions} reviewer decision(s) -> "
            f"{self.n_labelled_pairs} labelled pairs",
            f"Threshold {self.threshold_before:.3f} -> {self.threshold_after:.3f}",
            f"Precision {self.precision_before:.3f} -> {self.precision_after:.3f}",
            f"Recall    {self.recall_before:.3f} -> {self.recall_after:.3f}",
            f"F1        {self.f1_before:.3f} -> {self.f1_after:.3f} "
            f"({self.delta_f1():+.3f})",
        ]


# ---------------------------------------------------------------------------
# Queue construction
# ---------------------------------------------------------------------------
def build_queue(
    clusters,
    threshold: float,
    tiers: tuple[str, ...] = ("MEDIUM", "UNKNOWN"),
    limit: int | None = None,
) -> list[ReviewItem]:
    """Build the review queue, ordered by uncertainty rather than by score.

    Sorting by score descending would show the reviewer the cases the model is
    already sure about, which teaches it nothing. Uncertainty sampling shows the
    cases nearest the decision boundary first, where a human label moves the
    threshold most -- the same principle behind classical active learning.

    Args:
        clusters: Clusters from ``matching_engine.run_matching``.
        threshold: Current decision boundary.
        tiers: Which tiers require review.
        limit: Optional cap on queue length.

    Returns:
        Review items, most informative first.
    """
    items: list[ReviewItem] = []
    for position, item in enumerate(c for c in clusters if c.tier in tiers):
        items.append(
            ReviewItem(
                item_id=position,
                cluster_id=item.cluster_id,
                members=list(item.members),
                tier=item.tier,
                score=item.min_score,
                uncertainty=abs(item.min_score - threshold),
                cpses=list(item.cpses),
                reason=item.tier_reason,
            )
        )
    items.sort(key=lambda i: i.uncertainty)
    for new_id, item in enumerate(items):
        item.item_id = new_id
    return items[:limit] if limit else items


# ---------------------------------------------------------------------------
# Decision recording
# ---------------------------------------------------------------------------
def record_decision(
    item: ReviewItem,
    action: str,
    reviewer: str,
    comment: str = "",
    corrected_members: list[int] | None = None,
    log_path: Path | None = None,
) -> DecisionRecord:
    """Record a reviewer decision and append it to the durable log.

    Args:
        item: The queue item being decided.
        action: One of approve, reject, edit.
        reviewer: Identity of the reviewer.
        comment: Optional free-text justification.
        corrected_members: Required for ``edit`` -- the corrected membership.
        log_path: Destination log; defaults to ``config.REVIEW_LOG_PATH``.

    Returns:
        The persisted :class:`DecisionRecord`.

    Raises:
        ValueError: On an unknown action, a missing reviewer identity, or an
            edit without corrected membership.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action {action!r}; expected one of {VALID_ACTIONS}.")
    if not reviewer or not reviewer.strip():
        raise ValueError(
            "Reviewer identity is required: an unattributed decision cannot be "
            "audited or rolled back to a responsible party."
        )
    if action == EDIT and not corrected_members:
        raise ValueError(
            "An 'edit' decision must supply corrected_members; otherwise it is "
            "indistinguishable from an approval."
        )

    record = DecisionRecord(
        item_id=item.item_id,
        cluster_id=item.cluster_id,
        action=action,
        reviewer=reviewer.strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        comment=comment,
        members=list(corrected_members) if action == EDIT else list(item.members),
        original_members=list(item.members),
        score=item.score,
    )
    append_decision(record, log_path)
    return record


def append_decision(record: DecisionRecord, log_path: Path | None = None) -> None:
    """Append one decision to the JSONL log.

    Args:
        record: The decision.
        log_path: Destination; defaults to ``config.REVIEW_LOG_PATH``.
    """
    log_path = log_path or config.REVIEW_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def load_decisions(log_path: Path | None = None) -> list[DecisionRecord]:
    """Read every decision from the log.

    Args:
        log_path: Source; defaults to ``config.REVIEW_LOG_PATH``.

    Returns:
        Decisions in the order they were made. Empty when no log exists.
    """
    log_path = log_path or config.REVIEW_LOG_PATH
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(DecisionRecord(**json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Active learning
# ---------------------------------------------------------------------------
def decisions_to_labels(
    decisions: list[DecisionRecord],
) -> dict[tuple[int, int], int]:
    """Flatten reviewer decisions into pair labels.

    Later decisions win, so a reviewer correcting an earlier call overrides it
    rather than adding a contradictory duplicate.

    Args:
        decisions: Decisions in chronological order.

    Returns:
        ``{(idx_a, idx_b): label}``.
    """
    labels: dict[tuple[int, int], int] = {}
    for decision in decisions:
        for a, b, label in decision.labelled_pairs():
            labels[(a, b)] = label
    return labels


def apply_decisions(clusters, decisions: list[DecisionRecord]):
    """Apply reviewer decisions directly to the cluster set.

    This is the deterministic half of the feedback loop, and it is what makes
    review worth doing immediately rather than only after a retrain: an approved
    cluster is promoted, a rejected one is dropped, and an edited one is
    corrected to the membership the reviewer specified. The human's judgement is
    applied as fact, not as a training hint the model may or may not absorb.

    Args:
        clusters: Clusters from ``matching_engine.run_matching``.
        decisions: Reviewer decisions, chronological. Later decisions on the
            same cluster win.

    Returns:
        A new cluster list with decisions applied.
    """
    import copy

    latest: dict[int, DecisionRecord] = {}
    for decision in decisions:
        latest[decision.cluster_id] = decision

    applied = []
    for item in clusters:
        decision = latest.get(item.cluster_id)
        if decision is None:
            applied.append(item)
            continue
        if decision.action == REJECT:
            continue

        revised = copy.deepcopy(item)
        if decision.action == EDIT:
            kept = set(decision.members)
            revised.members = sorted(kept)
            revised.edges = [
                (a, b, w) for a, b, w in revised.edges if a in kept and b in kept
            ]
            if len(revised.members) < 2:
                continue
            weights = [w for _, _, w in revised.edges] or [revised.min_score]
            revised.mean_score = float(np.mean(weights))
            revised.min_score = float(np.min(weights))

        revised.tier = "HIGH"
        revised.tier_reason = (
            f"Confirmed by {decision.reviewer} at {decision.timestamp} "
            f"({decision.action})."
        )
        applied.append(revised)
    return applied


def recalibrate_threshold(
    scored_pairs: pd.DataFrame,
    decisions: list[DecisionRecord],
    current_threshold: float,
    score_column: str = "match_probability",
) -> float:
    """Move the decision threshold to fit accumulated reviewer labels.

    CAVEAT, and it is a real one: the review queue is filled by uncertainty
    sampling, so reviewer-labelled pairs are drawn almost entirely from the
    neighbourhood of the current boundary. Fitting a global threshold to that
    sample is statistically invalid -- the sample is not representative of the
    score distribution, and doing it naively on this dataset drags the threshold
    from 0.55 to 0.19 and costs ~2 points of F1.

    The function is therefore blended rather than replaced: it moves the
    threshold only part of the way toward the review-fitted value, and only once
    enough labels exist for the fit to mean anything. Use
    :func:`apply_decisions` and :func:`retrain_with_feedback` as the primary
    feedback path; this is a slow secondary correction.

    Args:
        scored_pairs: Scored pairs.
        decisions: Reviewer decisions.
        current_threshold: Threshold in force.
        score_column: Score to threshold on.

    Returns:
        The recalibrated threshold, or ``current_threshold`` when there are too
        few labels or only one class among them to fit anything meaningful.
    """
    labels = decisions_to_labels(decisions)
    if len(labels) < config.MIN_REVIEW_LABELS_FOR_RECALIBRATION:
        return current_threshold

    keyed = scored_pairs.set_index(["idx_a", "idx_b"])
    scores, targets = [], []
    for (a, b), label in labels.items():
        if (a, b) in keyed.index:
            scores.append(float(keyed.at[(a, b), score_column]))
            targets.append(label)

    if len(set(targets)) < 2:
        return current_threshold

    from .classifier import tune_threshold

    fitted, _ = tune_threshold(np.array(targets), np.array(scores))
    blend = config.REVIEW_RECALIBRATION_BLEND
    return (1 - blend) * current_threshold + blend * fitted


def retrain_with_feedback(
    scored_pairs: pd.DataFrame,
    eval_df: pd.DataFrame,
    decisions: list[DecisionRecord],
    feedback_weight: int = 5,
):
    """Refit the classifier using the reviewer's labels for reviewed pairs.

    The reviewer's verdict OVERRIDES the label those pairs would otherwise
    carry. This matters more than it looks: an earlier version merely duplicated
    reviewed rows while leaving their labels derived from ``GroundTruth_Group``,
    which meant the "feedback loop" fed the model nothing the model had not
    already been trained on. In production there is no ground-truth column at
    all -- the reviewer *is* the label -- so the override is the whole
    mechanism, not a refinement of it.

    Reviewed pairs are repeated ``feedback_weight`` times, because a few dozen
    human decisions against ~60,000 training rows would otherwise change nothing
    measurable. The weighting is explicit and tunable rather than buried in a
    loss function.

    MEASURED RESULT, stated plainly: on this dataset retraining does NOT yet
    help. Across 60 simulated decisions, applying them deterministically
    (:func:`apply_decisions`) lifts cluster F1 from 0.897 to 0.921, while
    additionally retraining moves it to 0.891 -- slightly worse than not
    retraining at all. The cause is the same uncertainty sampling that makes the
    queue efficient: every reviewed pair sits near the boundary, and upweighting
    a boundary-only sample five-fold biases the model toward that region without
    adding information it lacked. Retraining should pay off once decisions
    accumulate into the thousands and cover more of the distribution; at 60 it
    does not, and reporting it as a gain would be false.

    The demo therefore uses :func:`apply_decisions` as the primary feedback
    path, and this function is available and instrumented rather than switched
    on by default.

    Args:
        scored_pairs: Scored pairs.
        eval_df: Evaluation frame, used for the held-out split and for pairs no
            reviewer has touched.
        decisions: Reviewer decisions.
        feedback_weight: Repetition count for reviewer-labelled rows.

    Returns:
        ``(pipeline, report)`` as from ``classifier.train``.
    """
    from .classifier import train

    labels = decisions_to_labels(decisions)
    if not labels:
        return train(scored_pairs, eval_df)

    keys = list(zip(scored_pairs["idx_a"], scored_pairs["idx_b"]))
    reviewed_mask = np.array([key in labels for key in keys])

    augmented = scored_pairs.copy()
    augmented["reviewer_label"] = [labels.get(key, -1) for key in keys]

    repeated = pd.concat(
        [augmented] + [augmented[reviewed_mask]] * max(0, feedback_weight - 1),
        ignore_index=True,
    )
    repeated.attrs.update(scored_pairs.attrs)
    return train(repeated, eval_df, reviewer_label_column="reviewer_label")


def measure_improvement(
    before: dict[str, float],
    after: dict[str, float],
    n_decisions: int,
    n_labelled_pairs: int,
    threshold_before: float,
    threshold_after: float,
) -> ActiveLearningReport:
    """Assemble the before/after comparison for the app.

    Both sides must come from the same evaluation set; otherwise the comparison
    measures the yardstick, not the model.

    Args:
        before: Metrics before feedback, with keys precision/recall/f1.
        after: Metrics after feedback, same keys.
        n_decisions: Decisions applied.
        n_labelled_pairs: Pairs those decisions labelled.
        threshold_before: Threshold before recalibration.
        threshold_after: Threshold after.

    Returns:
        A populated :class:`ActiveLearningReport`.
    """
    return ActiveLearningReport(
        n_decisions=n_decisions,
        n_labelled_pairs=n_labelled_pairs,
        threshold_before=threshold_before,
        threshold_after=threshold_after,
        precision_before=before.get("precision", 0.0),
        precision_after=after.get("precision", 0.0),
        recall_before=before.get("recall", 0.0),
        recall_after=after.get("recall", 0.0),
        f1_before=before.get("f1", 0.0),
        f1_after=after.get("f1", 0.0),
    )


def simulate_reviews(
    queue: list[ReviewItem],
    eval_df: pd.DataFrame,
    n: int = 25,
    reviewer: str = "sim.reviewer",
    log_path: Path | None = None,
) -> list[DecisionRecord]:
    """Generate decisions from ground truth, to demonstrate the loop.

    This exists so the active-learning improvement can be shown in a live demo
    without a human clicking through 25 clusters. It is a simulation of a
    reviewer, clearly named as one, and it must never be presented as evidence
    that real reviewers behave this way -- a real reviewer is slower, less
    consistent, and occasionally right when the ground truth is wrong.

    Args:
        queue: Review queue, most uncertain first.
        eval_df: Evaluation frame supplying the "correct" answer.
        n: How many items to decide.
        reviewer: Identity to attribute the simulated decisions to.
        log_path: Destination log.

    Returns:
        The simulated decisions.
    """
    groups = eval_df["GroundTruth_Group"]
    decisions: list[DecisionRecord] = []

    for item in queue[:n]:
        member_groups = [groups.iat[i] for i in item.members]
        majority = max(set(member_groups), key=member_groups.count)
        correct = [i for i, g in zip(item.members, member_groups) if g == majority]

        if len(correct) == len(item.members):
            action, corrected = APPROVE, None
        elif len(correct) < 2:
            action, corrected = REJECT, None
        else:
            action, corrected = EDIT, correct

        decisions.append(
            record_decision(
                item,
                action,
                reviewer=reviewer,
                comment="Simulated decision derived from ground truth.",
                corrected_members=corrected,
                log_path=log_path,
            )
        )
    return decisions


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    print("Run via app.py or notebooks/evaluation.ipynb.")
