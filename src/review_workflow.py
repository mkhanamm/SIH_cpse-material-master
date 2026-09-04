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
