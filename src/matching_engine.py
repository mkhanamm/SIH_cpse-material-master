"""
Clustering into equivalence groups and confidence routing (Spec section 4.6).

WHAT THIS FILE DOES
    Turns scored pairs into material equivalence groups, then routes each group
    to an outcome tier.

    Clustering: a graph over materials, edges = pairs above the match threshold,
    equivalence groups = connected components. Because naive connected
    components can chain (A~B, B~C, A!~C) into one oversized cluster, a cohesion
    check splits any component whose internal average score falls below
    config.MIN_CLUSTER_COHESION.

    Confidence tiers (thresholds in config.py):
      HIGH   - auto-suggest a Common National Material Code.
      MEDIUM - queue for human review (this is the review_workflow input).
      LOW    - no match asserted.
      UNKNOWN- too many missing attributes to decide; explicitly flagged
               "insufficient data, needs manual review" rather than guessed.

INPUTS
    Scored/classified pair table.

OUTPUTS
    MatchResult - .clusters (list of MaterialCluster), .tier_counts
    MaterialCluster - members, mean/min internal score, tier, evidence refs

KEY FUNCTIONS
    build_graph(scored_pairs, threshold) -> networkx.Graph
    cluster(graph)                       -> list[MaterialCluster]
    assign_tiers(clusters)               -> list[MaterialCluster]
    evaluate(clusters, truth_pairs)      -> dict  (precision/recall/F1,
                                            within- vs cross-CPSE breakdown)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from . import config

HIGH, MEDIUM, LOW, UNKNOWN = "HIGH", "MEDIUM", "LOW", "UNKNOWN"


@dataclass
class MaterialCluster:
    """A proposed equivalence group of material records.

    Attributes:
        cluster_id: Stable identifier within one matching run.
        members: Record indices in the group.
        mean_score: Average score over the group's internal edges.
        min_score: Weakest internal edge -- the honest measure of how tightly
            the group holds together, since one weak link is what a chained
            cluster is made of.
        tier: HIGH / MEDIUM / LOW / UNKNOWN.
        cpses: Distinct CPSEs represented.
        is_cross_cpse: Whether the group spans more than one CPSE.
        min_known_attrs: Fewest known attributes among members; drives the
            UNKNOWN routing.
        edges: Internal edges with their scores, for the explanation layer.
        tier_reason: Plain-language justification for the tier assigned.
    """

    cluster_id: int
    members: list[int]
    mean_score: float
    min_score: float
    tier: str = LOW
    cpses: list[str] = field(default_factory=list)
    is_cross_cpse: bool = False
    min_known_attrs: int = 0
    edges: list[tuple[int, int, float]] = field(default_factory=list)
    tier_reason: str = ""

    @property
    def size(self) -> int:
        """Number of records in the cluster.

        Returns:
            Member count.
        """
        return len(self.members)

    def pairs(self) -> list[tuple[int, int]]:
        """Enumerate every within-cluster pair.

        Clustering asserts that all members are the same material, so the
        cluster claims every internal pair -- including pairs that were never
        directly scored. Evaluation must charge for those transitive claims.

        Returns:
            Sorted index pairs.
        """
        return list(itertools.combinations(sorted(self.members), 2))


@dataclass
class MatchResult:
    """Output of a full matching run.

    Attributes:
        clusters: Multi-member clusters only; singletons are not materials
            "matched to themselves" and would distort every count.
        tier_counts: Cluster count per tier.
        threshold: Edge threshold used.
        n_records: Records considered.
        n_clustered_records: Records that ended up in a multi-member cluster.
        n_split_by_cohesion: Components broken up by the cohesion check.
    """

    clusters: list[MaterialCluster]
    tier_counts: dict[str, int]
    threshold: float
    n_records: int
    n_clustered_records: int
    n_split_by_cohesion: int = 0

    def by_tier(self, tier: str) -> list[MaterialCluster]:
        """Select clusters in one confidence tier.

        Args:
            tier: HIGH, MEDIUM, LOW or UNKNOWN.

        Returns:
            Matching clusters.
        """
        return [c for c in self.clusters if c.tier == tier]


# ---------------------------------------------------------------------------
# Graph construction and clustering
# ---------------------------------------------------------------------------
def build_graph(
    scored_pairs: pd.DataFrame,
    threshold: float,
    score_column: str = "match_probability",
) -> nx.Graph:
    """Build the match graph from pairs scoring at or above the threshold.

    Args:
        scored_pairs: Scored pairs, carrying ``idx_a``, ``idx_b`` and the score
            column.
        threshold: Minimum score for an edge.
        score_column: Which score to use -- ``match_probability`` from the
            trained classifier, or ``fused`` for the hand-weighted score.

    Returns:
        An undirected graph with ``weight`` on every edge.

    Raises:
        KeyError: If ``score_column`` is absent.
    """
    if score_column not in scored_pairs.columns:
        raise KeyError(
            f"{score_column!r} not in scored pairs. Run classifier.predict_proba "
            "and assign it, or pass score_column='fused'."
        )

    kept = scored_pairs[scored_pairs[score_column] >= threshold]
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        zip(kept["idx_a"], kept["idx_b"], kept[score_column])
    )
    return graph


def _cohesion(graph: nx.Graph, nodes: list[int]) -> float:
    """Average edge weight within a set of nodes.

    Args:
        graph: The match graph.
        nodes: Node subset.

    Returns:
        Mean weight over present internal edges, or 0.0 if there are none.
    """
    subgraph = graph.subgraph(nodes)
    weights = [d["weight"] for _, _, d in subgraph.edges(data=True)]
    return float(np.mean(weights)) if weights else 0.0


def _split_low_cohesion(
    graph: nx.Graph, nodes: list[int], cohesion_floor: float
) -> list[list[int]]:
    """Break a loosely-connected component into tighter sub-clusters.

    Connected components are transitive, but material equivalence is not: if A
    matches B on size and B matches C on grade, A and C may share nothing. One
    chain of weak edges can otherwise swallow an entire category into a single
    national code -- the most damaging failure mode this system has, because it
    silently merges materials that must stay distinct.

    The split removes the weakest edges until the component separates, then
    recurses on each part.

    Args:
        graph: The match graph.
        nodes: Component members.
        cohesion_floor: Minimum mean internal weight for a component to stand.

    Returns:
        A list of node groups, each meeting the cohesion floor (or too small to
        split further).
    """
    if len(nodes) <= 2 or _cohesion(graph, nodes) >= cohesion_floor:
        return [nodes]

    subgraph = graph.subgraph(nodes).copy()
    edges_by_weight = sorted(
        subgraph.edges(data=True), key=lambda e: e[2]["weight"]
    )
    for u, v, _ in edges_by_weight:
        subgraph.remove_edge(u, v)
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            parts: list[list[int]] = []
            for component in components:
                parts.extend(
                    _split_low_cohesion(graph, sorted(component), cohesion_floor)
                )
            return parts
    return [nodes]


def cluster(
    graph: nx.Graph,
    df: pd.DataFrame,
    attribute_counts: pd.Series | None = None,
    cohesion_floor: float = 0.0,
) -> tuple[list[MaterialCluster], int]:
    """Extract equivalence groups from the match graph.

    Args:
        graph: Output of :func:`build_graph`.
        df: Frame carrying ``CPSE`` for cross-CPSE detection.
        attribute_counts: Known-attribute count per record, for tier routing.
        cohesion_floor: Minimum mean internal edge weight for a component to
            survive intact; below it the component is split.

    Returns:
        ``(clusters, n_split_by_cohesion)``. Only multi-member clusters are
        returned.
    """
    clusters: list[MaterialCluster] = []
    n_split = 0
    cluster_id = 0

    for component in nx.connected_components(graph):
        nodes = sorted(component)
        parts = _split_low_cohesion(graph, nodes, cohesion_floor)
        if len(parts) > 1:
            n_split += 1

        for part in parts:
            if len(part) < 2:
                continue
            subgraph = graph.subgraph(part)
            edges = [
                (u, v, float(d["weight"])) for u, v, d in subgraph.edges(data=True)
            ]
            weights = [w for _, _, w in edges] or [0.0]
            cpses = sorted({str(df.at[i, "CPSE"]) for i in part})
            known = (
                int(attribute_counts.reindex(part).min())
                if attribute_counts is not None
                else 0
            )

            clusters.append(
                MaterialCluster(
                    cluster_id=cluster_id,
                    members=part,
                    mean_score=float(np.mean(weights)),
                    min_score=float(np.min(weights)),
                    cpses=cpses,
                    is_cross_cpse=len(cpses) > 1,
                    min_known_attrs=known,
                    edges=edges,
                )
            )
            cluster_id += 1

    return clusters, n_split


# ---------------------------------------------------------------------------
# Confidence routing
# ---------------------------------------------------------------------------
def assign_tiers(
    clusters: list[MaterialCluster], high: float, medium: float
) -> list[MaterialCluster]:
    """Route each cluster to a confidence tier, in place.

    A cluster is promoted to HIGH only if its *weakest* internal edge clears the
    high threshold -- the mean would let one strong pair carry a doubtful third
    member into an auto-approved national code.

    Attribute depth gates promotion independently of score. A cluster whose
    poorest-documented member states fewer than
    ``config.MIN_KNOWN_ATTRIBUTES_FOR_AUTO`` attributes goes to UNKNOWN
    regardless of text similarity: with 4,650 of 5,008 records missing an
    operating parameter and 4,342 missing a specification, a high text score on
    two bare descriptions is not evidence of equivalence, it is evidence of two
    equally sparse descriptions.

    Args:
        clusters: Clusters from :func:`cluster`.
        high: Auto-approve cut-off, from ``classifier.calibrate_tiers``.
        medium: Human-review cut-off, from the same calibration. Both are passed
            in rather than read from config, because their correct values depend
            on which scoring scale produced the edge weights.

    Returns:
        The same list, with ``tier`` and ``tier_reason`` populated.
    """
    for item in clusters:
        if item.min_known_attrs < config.MIN_KNOWN_ATTRIBUTES_FOR_AUTO:
            item.tier = UNKNOWN
            item.tier_reason = (
                f"Insufficient data: poorest-documented member states only "
                f"{item.min_known_attrs} structured attribute(s); "
                f"{config.MIN_KNOWN_ATTRIBUTES_FOR_AUTO} required for automatic "
                "matching. Needs manual review."
            )
        elif item.min_score >= high:
            item.tier = HIGH
            item.tier_reason = (
                f"All {len(item.edges)} internal comparison(s) at or above the "
                f"auto-approve cut-off {high:.3f} (weakest {item.min_score:.3f})."
            )
        elif item.min_score >= medium:
            item.tier = MEDIUM
            item.tier_reason = (
                f"Weakest internal comparison {item.min_score:.3f} sits between "
                f"the review cut-off {medium:.3f} and the auto-approve cut-off "
                f"{high:.3f}. Queued for human review."
            )
        else:
            item.tier = LOW
            item.tier_reason = (
                f"Weakest internal comparison {item.min_score:.3f} is below the "
                f"review cut-off {medium:.3f}. No match asserted."
            )
    return clusters


def tune_edge_threshold(
    scored_pairs: pd.DataFrame,
    df: pd.DataFrame,
    truth_pairs: set[tuple[int, int]],
    validation_records: set[int],
    attribute_counts: pd.Series | None = None,
    score_column: str = "match_probability",
    candidates: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose the graph edge threshold by CLUSTER-level F1 on validation data.

    The pair-optimal threshold is the wrong threshold for clustering, and by a
    wide margin. A cluster asserts equivalence between every pair of its
    members, so one spurious edge joining two correct clusters of size 5
    manufactures 25 false pairs. Errors are amplified transitively; the
    threshold must be stricter than pair-level F1 would suggest.

    Measured on this dataset the pair-optimal cut-off is ~0.06 while the
    cluster-optimal is ~0.70 -- an order of magnitude apart. Tuning once at the
    pair level and reusing the number is a quiet, plausible-looking way to lose
    most of the system's precision.

    Two subtleties, both of which cost real precision when got wrong:

    1. The graph is built over the WHOLE corpus, not just the tuning records.
       Chaining risk is a function of graph density, and clustering a 15%
       subsample makes the corpus look far sparser than it is: tuning on a
       subsample-only graph selected 0.15 here, which delivers 0.44 precision at
       full scale. Density is an unlabelled property of the data, available at
       inference time, so using it leaks nothing.

    2. The evaluated pairs are those with both endpoints OUTSIDE the test split
       -- i.e. train and validation records together -- not validation alone.
       False positives arise from cross-group pairs, and with only 15% of groups
       held out, two validation groups rarely chain into the same cluster. The
       validation-only sample is therefore starved of precisely the errors it
       needs to measure, and reports 0.86 precision where the true figure is
       0.55. Test labels are never touched either way.

    Args:
        scored_pairs: Scored candidate pairs.
        df: Frame carrying ``CPSE``.
        truth_pairs: Ground-truth duplicate pairs.
        validation_records: Record indices available for tuning -- pass every
            record outside the test split, for the reason given above.
        attribute_counts: Known-attribute count per record.
        score_column: Score to threshold on.
        candidates: Thresholds to try; defaults to 0.10-0.95 in steps of 0.05.

    Returns:
        ``(best_threshold, sweep_table)``. The table is kept for the report --
        it is the evidence that the choice was measured, not assumed.
    """
    candidates = (
        candidates if candidates is not None else np.arange(0.10, 0.96, 0.05)
    )

    subset_truth = {
        (a, b)
        for a, b in truth_pairs
        if a in validation_records and b in validation_records
    }

    rows = []
    best_threshold, best_f1 = float(candidates[0]), -1.0

    for threshold in candidates:
        graph = build_graph(scored_pairs, float(threshold), score_column)
        clusters, n_split = cluster(
            graph, df, attribute_counts, cohesion_floor=float(threshold)
        )
        clusters = assign_tiers(
            clusters, high=float(threshold), medium=float(threshold)
        )

        predicted = set()
        for item in clusters:
            if item.tier == LOW:
                continue
            predicted.update(
                (a, b)
                for a, b in item.pairs()
                if a in validation_records and b in validation_records
            )

        true_positives = len(predicted & subset_truth)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(subset_truth) if subset_truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "n_clusters": len(clusters),
                "n_split": n_split,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1

    return best_threshold, pd.DataFrame(rows)


def calibrate_tiers_from_sweep(
    sweep: pd.DataFrame,
    edge_threshold: float,
    precision_target: float = config.AUTO_APPROVE_PRECISION_TARGET,
):
    """Derive HIGH/MEDIUM cut-offs from the cluster-level threshold sweep.

    The pair-level calibration in ``classifier.calibrate_tiers`` answers "how
    precise is this pair judgement?", which is the wrong question for tiering: a
    cluster is what gets a national code, and a cluster's precision is not its
    weakest pair's precision. Calibrating tiers on the cluster sweep also keeps
    the cut-offs on the same scale as the edge weights, so the HIGH bar cannot
    end up *below* the edge threshold and collapse every cluster into one tier
    -- which is exactly what pair-level calibration did here, leaving the review
    queue empty.

    MEDIUM is the edge threshold itself: below it nothing enters the graph, so
    nothing can be reviewed. HIGH is the lowest swept threshold meeting the
    auto-approve precision target. The target is deliberately stricter than the
    F1-optimal point, because merging two CPSEs' material codes without a human
    is a high-consequence action and F1 weighs a missed duplicate the same as a
    wrongly merged one.

    Args:
        sweep: Table from :func:`tune_edge_threshold`.
        edge_threshold: Chosen edge threshold, used as the MEDIUM cut-off.
        precision_target: Cluster-level precision required for auto-approval.

    Returns:
        A ``classifier.TierThresholds``. When no swept threshold meets the
        target, the most precise available point is used and its achieved
        precision reported rather than the target being quietly relaxed.
    """
    from .classifier import TierThresholds

    qualifying = sweep[
        (sweep["precision"] >= precision_target)
        & (sweep["threshold"] >= edge_threshold)
    ]
    if len(qualifying):
        row = qualifying.iloc[0]
    else:
        row = sweep.loc[sweep["precision"].idxmax()]

    return TierThresholds(
        high=float(row["threshold"]),
        medium=float(edge_threshold),
        high_precision_target=precision_target,
        achieved_high_precision=float(row["precision"]),
        achieved_high_recall=float(row["recall"]),
    )


def run_matching(
    scored_pairs: pd.DataFrame,
    df: pd.DataFrame,
    tiers,
    edge_threshold: float | None = None,
    score_column: str = "match_probability",
    attribute_counts: pd.Series | None = None,
) -> MatchResult:
    """Run graph construction, clustering and tier routing end to end.

    Args:
        scored_pairs: Scored (and optionally classified) candidate pairs.
        df: Frame carrying ``CPSE``.
        tiers: ``classifier.TierThresholds`` from validation calibration.
        edge_threshold: Cut-off for admitting an edge to the match graph. This
            is a *separate* knob from the tier cut-offs and should come from
            :func:`tune_edge_threshold`; see that function for why the
            pair-optimal value is unsuitable here. Falls back to the HIGH tier
            cut-off, which is conservative, rather than to MEDIUM.
        score_column: Score to threshold on.
        attribute_counts: Known-attribute count per record.

    Returns:
        A populated :class:`MatchResult`.
    """
    edge_threshold = tiers.high if edge_threshold is None else edge_threshold
    cohesion_floor = edge_threshold * config.MIN_CLUSTER_COHESION_RATIO
    graph = build_graph(scored_pairs, edge_threshold, score_column)
    clusters, n_split = cluster(graph, df, attribute_counts, cohesion_floor)
    clusters = assign_tiers(clusters, tiers.high, tiers.medium)

    tier_counts = {tier: 0 for tier in (HIGH, MEDIUM, LOW, UNKNOWN)}
    for item in clusters:
        tier_counts[item.tier] += 1

    return MatchResult(
        clusters=clusters,
        tier_counts=tier_counts,
        threshold=edge_threshold,
        n_records=len(df),
        n_clustered_records=sum(c.size for c in clusters),
        n_split_by_cohesion=n_split,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    result: MatchResult,
    truth_pairs: set[tuple[int, int]],
    df: pd.DataFrame,
    tiers: tuple[str, ...] = (HIGH, MEDIUM),
) -> dict[str, float]:
    """Score the clustering against ground truth at the pair level.

    Cluster-level rather than edge-level evaluation is the honest measure: a
    cluster asserts equivalence between *every* pair of its members, including
    pairs the matcher never directly compared. Those transitive claims are
    charged here.

    Args:
        result: Output of :func:`run_matching`.
        truth_pairs: Ground-truth duplicate pairs.
        df: Frame carrying ``CPSE``.
        tiers: Which tiers count as an asserted match. Defaults to HIGH and
            MEDIUM, i.e. everything the system either auto-approves or puts in
            front of a reviewer.

    Returns:
        Dict of precision/recall/F1 overall and split cross-CPSE vs
        within-CPSE, plus raw counts.
    """
    predicted: set[tuple[int, int]] = set()
    for item in result.clusters:
        if item.tier in tiers:
            predicted.update(item.pairs())

    true_positives = predicted & truth_pairs
    cpse = df["CPSE"]

    def _split(pairs: set[tuple[int, int]], cross: bool) -> set[tuple[int, int]]:
        return {
            (a, b) for a, b in pairs if (cpse.at[a] != cpse.at[b]) == cross
        }

    def _prf(pred: set, truth: set) -> dict[str, float]:
        tp = len(pred & truth)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(truth) if truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "predicted": len(pred),
            "actual": len(truth),
        }

    overall = _prf(predicted, truth_pairs)
    cross = _prf(_split(predicted, True), _split(truth_pairs, True))
    within = _prf(_split(predicted, False), _split(truth_pairs, False))

    return {
        "tiers_counted": ", ".join(tiers),
        **{f"overall_{k}": v for k, v in overall.items()},
        **{f"cross_cpse_{k}": v for k, v in cross.items()},
        **{f"within_cpse_{k}": v for k, v in within.items()},
        "n_clusters": len(result.clusters),
        "n_clustered_records": result.n_clustered_records,
        "n_split_by_cohesion": result.n_split_by_cohesion,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from . import attribute_extraction, blocking, classifier, ingestion, similarity
    from .normalization import normalize_series

    dataset = ingestion.load_dataset()
    extracted = attribute_extraction.extract_frame(dataset.pipeline_df)
    joined = dataset.pipeline_df.join(extracted)
    text = normalize_series(joined[config.INPUT_TEXT_COLUMN])

    candidates = blocking.candidate_pairs(joined)
    scored = similarity.score_pairs(joined, candidates.pairs, text)
    model, report = classifier.train(scored, dataset.eval_df)
    scored["match_probability"] = classifier.predict_proba(model, scored)
    truth = ingestion.ground_truth_pairs(dataset.eval_df)

    train_mask, val_mask, test_mask, _ = classifier.group_disjoint_split(
        scored, dataset.eval_df
    )
    test_records = set(scored.loc[test_mask, "idx_a"]) | set(
        scored.loc[test_mask, "idx_b"]
    )
    tuning_records = set(joined.index) - test_records
    edge_threshold, sweep = tune_edge_threshold(
        scored, joined, truth, tuning_records, extracted["n_known_attributes"]
    )
    print("\nedge threshold sweep (non-test records; test labels untouched):")
    print(sweep.to_string(index=False))
    print(f"chosen edge threshold: {edge_threshold:.2f}\n")

    tiers = calibrate_tiers_from_sweep(sweep, edge_threshold)
    print("\n".join(tiers.summary_lines()))
    print()

    result = run_matching(
        scored,
        joined,
        tiers=tiers,
        edge_threshold=edge_threshold,
        attribute_counts=extracted["n_known_attributes"],
    )
    print("tier counts:", result.tier_counts)
    for label, tiers_counted in (
        ("SURFACED (HIGH+MEDIUM+UNKNOWN)", (HIGH, MEDIUM, UNKNOWN)),
        ("AUTO-APPROVED (HIGH only)", (HIGH,)),
    ):
        metrics = evaluate(result, truth, joined, tiers=tiers_counted)
        print(f"\n{label}")
        for scope in ("overall", "cross_cpse", "within_cpse"):
            print(
                f"  {scope:12s} P {metrics[f'{scope}_precision']:.3f} "
                f"R {metrics[f'{scope}_recall']:.3f} "
                f"F1 {metrics[f'{scope}_f1']:.3f} "
                f"(TP {metrics[f'{scope}_true_positives']}/"
                f"{metrics[f'{scope}_actual']})"
            )
