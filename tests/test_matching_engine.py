"""Unit tests for src.matching_engine -- clustering, cohesion splitting, tier assignment boundaries."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from src import config
from src.classifier import fallback_tiers
from src.matching_engine import (
    HIGH,
    LOW,
    MEDIUM,
    UNKNOWN,
    MaterialCluster,
    MatchResult,
    assign_tiers,
    build_graph,
    cluster,
    evaluate,
    run_matching,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Four records across three CPSEs."""
    return pd.DataFrame(
        {"CPSE": ["ONGC", "BPCL", "IOCL", "ONGC"], "Material Category": ["Valve"] * 4}
    )


@pytest.fixture
def rich_attributes() -> pd.Series:
    """Attribute counts high enough to clear the auto-match gate."""
    return pd.Series([5, 5, 5, 5])


class TestGraphConstruction:
    def test_only_edges_at_or_above_threshold_kept(self):
        pairs = pd.DataFrame(
            {"idx_a": [0, 1], "idx_b": [1, 2], "match_probability": [0.9, 0.3]}
        )
        graph = build_graph(pairs, threshold=0.5)
        assert graph.has_edge(0, 1)
        assert not graph.has_edge(1, 2)

    def test_threshold_is_inclusive(self):
        pairs = pd.DataFrame(
            {"idx_a": [0], "idx_b": [1], "match_probability": [0.5]}
        )
        assert build_graph(pairs, threshold=0.5).has_edge(0, 1)

    def test_missing_score_column_raises_clearly(self):
        pairs = pd.DataFrame({"idx_a": [0], "idx_b": [1], "fused": [0.9]})
        with pytest.raises(KeyError, match="match_probability"):
            build_graph(pairs, threshold=0.5)

    def test_can_use_fused_score_instead(self):
        pairs = pd.DataFrame({"idx_a": [0], "idx_b": [1], "fused": [0.9]})
        assert build_graph(pairs, 0.5, score_column="fused").has_edge(0, 1)


class TestClustering:
    def test_connected_records_form_one_cluster(self, frame, rich_attributes):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.95), (1, 2, 0.92)])
        clusters, _ = cluster(graph, frame, rich_attributes, cohesion_floor=0.5)
        assert len(clusters) == 1
        assert clusters[0].members == [0, 1, 2]

    def test_singletons_are_not_clusters(self, frame, rich_attributes):
        """A record matched to nothing is not a cluster of one."""
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.95)])
        graph.add_node(3)
        clusters, _ = cluster(graph, frame, rich_attributes, cohesion_floor=0.5)
        assert all(c.size >= 2 for c in clusters)

    def test_weak_chain_is_split(self, frame, rich_attributes):
        """A~B strong, B~C weak: transitive chaining must not merge A and C."""
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.99), (1, 2, 0.30), (2, 3, 0.99)])
        clusters, n_split = cluster(
            graph, frame, rich_attributes, cohesion_floor=0.80
        )
        assert n_split == 1
        assert len(clusters) == 2
        members = sorted(sorted(c.members) for c in clusters)
        assert members == [[0, 1], [2, 3]]

    def test_cohesive_cluster_survives_intact(self, frame, rich_attributes):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.95), (1, 2, 0.94), (0, 2, 0.96)])
        clusters, n_split = cluster(
            graph, frame, rich_attributes, cohesion_floor=0.80
        )
        assert n_split == 0
        assert len(clusters) == 1

    def test_cross_cpse_detected(self, frame, rich_attributes):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.95)])
        clusters, _ = cluster(graph, frame, rich_attributes, cohesion_floor=0.5)
        assert clusters[0].is_cross_cpse
        assert clusters[0].cpses == ["BPCL", "ONGC"]

    def test_same_cpse_not_flagged_cross(self, frame, rich_attributes):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 3, 0.95)])
        clusters, _ = cluster(graph, frame, rich_attributes, cohesion_floor=0.5)
        assert not clusters[0].is_cross_cpse

    def test_min_score_is_the_weakest_edge(self, frame, rich_attributes):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.99), (1, 2, 0.85)])
        clusters, _ = cluster(graph, frame, rich_attributes, cohesion_floor=0.5)
        assert clusters[0].min_score == pytest.approx(0.85)


class TestTierRouting:
    def _cluster(self, min_score: float, known: int = 5) -> MaterialCluster:
        return MaterialCluster(
            cluster_id=0,
            members=[0, 1],
            mean_score=min_score,
            min_score=min_score,
            min_known_attrs=known,
            edges=[(0, 1, min_score)],
        )

    def test_high_tier(self):
        assigned = assign_tiers([self._cluster(0.95)], high=0.85, medium=0.55)
        assert assigned[0].tier == HIGH

    def test_medium_tier(self):
        assigned = assign_tiers([self._cluster(0.70)], high=0.85, medium=0.55)
        assert assigned[0].tier == MEDIUM

    def test_low_tier(self):
        assigned = assign_tiers([self._cluster(0.20)], high=0.85, medium=0.55)
        assert assigned[0].tier == LOW

    def test_boundaries_are_inclusive(self):
        assert assign_tiers([self._cluster(0.85)], 0.85, 0.55)[0].tier == HIGH
        assert assign_tiers([self._cluster(0.55)], 0.85, 0.55)[0].tier == MEDIUM

    def test_promotion_uses_weakest_edge_not_mean(self):
        """One strong pair must not carry a doubtful third member into auto-approval."""
        item = MaterialCluster(
            cluster_id=0,
            members=[0, 1, 2],
            mean_score=0.90,
            min_score=0.60,
            min_known_attrs=5,
        )
        assert assign_tiers([item], high=0.85, medium=0.55)[0].tier == MEDIUM

    def test_sparse_attributes_block_auto_match(self):
        """Approved design rule: text similarity alone cannot auto-approve."""
        sparse = self._cluster(0.99, known=config.MIN_KNOWN_ATTRIBUTES_FOR_AUTO - 1)
        assigned = assign_tiers([sparse], high=0.85, medium=0.55)
        assert assigned[0].tier == UNKNOWN
        assert "Insufficient data" in assigned[0].tier_reason

    def test_attribute_gate_outranks_score(self):
        sparse = self._cluster(1.0, known=0)
        assert assign_tiers([sparse], high=0.0, medium=0.0)[0].tier == UNKNOWN

    def test_every_tier_carries_a_reason(self):
        clusters = [self._cluster(s) for s in (0.95, 0.70, 0.20)]
        for item in assign_tiers(clusters, high=0.85, medium=0.55):
            assert item.tier_reason


class TestClusterPairs:
    def test_cluster_claims_every_internal_pair(self):
        """Including pairs never directly scored -- evaluation must charge for them."""
        item = MaterialCluster(
            cluster_id=0, members=[2, 0, 1], mean_score=0.9, min_score=0.9
        )
        assert item.pairs() == [(0, 1), (0, 2), (1, 2)]


class TestRunMatchingWithoutLabels:
    """The unlabelled path: no classifier, edge weight = fused score, fixed tiers."""

    def test_clusters_on_fused_score_alone(self, frame, rich_attributes):
        scored = pd.DataFrame(
            {
                "idx_a": [0, 1],
                "idx_b": [1, 2],
                "fused": [0.95, 0.92],
            }
        )
        tiers = fallback_tiers(high=0.85, medium=0.65)
        result = run_matching(
            scored,
            frame,
            tiers,
            edge_threshold=tiers.medium,
            score_column="fused",
            attribute_counts=rich_attributes,
        )
        assert len(result.clusters) == 1
        assert result.clusters[0].members == [0, 1, 2]

    def test_tiers_route_on_fused_score(self, frame, rich_attributes):
        scored = pd.DataFrame({"idx_a": [0], "idx_b": [1], "fused": [0.90]})
        tiers = fallback_tiers(high=0.85, medium=0.65)
        result = run_matching(
            scored, frame, tiers, edge_threshold=tiers.medium,
            score_column="fused", attribute_counts=rich_attributes,
        )
        assert result.clusters[0].tier == HIGH

    def test_below_medium_never_enters_the_graph(self, frame, rich_attributes):
        scored = pd.DataFrame({"idx_a": [0], "idx_b": [1], "fused": [0.40]})
        tiers = fallback_tiers(high=0.85, medium=0.65)
        result = run_matching(
            scored, frame, tiers, edge_threshold=tiers.medium,
            score_column="fused", attribute_counts=rich_attributes,
        )
        assert result.clusters == []

    def test_does_not_require_match_probability_column(self, frame, rich_attributes):
        """No classifier ran, so 'match_probability' need not exist at all."""
        scored = pd.DataFrame({"idx_a": [0], "idx_b": [1], "fused": [0.90]})
        assert "match_probability" not in scored.columns
        tiers = fallback_tiers()
        result = run_matching(
            scored, frame, tiers, edge_threshold=tiers.medium,
            score_column="fused", attribute_counts=rich_attributes,
        )
        assert len(result.clusters) == 1


class TestEvaluation:
    def test_perfect_clustering(self, frame):
        item = MaterialCluster(
            cluster_id=0, members=[0, 1], mean_score=0.9, min_score=0.9, tier=HIGH
        )
        result = MatchResult([item], {}, 0.5, 4, 2)
        metrics = evaluate(result, {(0, 1)}, frame)
        assert metrics["overall_precision"] == 1.0
        assert metrics["overall_recall"] == 1.0

    def test_transitive_false_positives_are_charged(self, frame):
        """A 3-member cluster covering one true pair claims three; precision is 1/3."""
        item = MaterialCluster(
            cluster_id=0, members=[0, 1, 2], mean_score=0.9, min_score=0.9, tier=HIGH
        )
        result = MatchResult([item], {}, 0.5, 4, 3)
        metrics = evaluate(result, {(0, 1)}, frame)
        assert metrics["overall_precision"] == pytest.approx(1 / 3)

    def test_excluded_tiers_do_not_count(self, frame):
        item = MaterialCluster(
            cluster_id=0, members=[0, 1], mean_score=0.9, min_score=0.9, tier=LOW
        )
        result = MatchResult([item], {}, 0.5, 4, 2)
        metrics = evaluate(result, {(0, 1)}, frame, tiers=(HIGH, MEDIUM))
        assert metrics["overall_recall"] == 0.0

    def test_cross_and_within_cpse_split(self, frame):
        clusters = [
            MaterialCluster(0, [0, 1], 0.9, 0.9, tier=HIGH),  # ONGC vs BPCL
            MaterialCluster(1, [0, 3], 0.9, 0.9, tier=HIGH),  # ONGC vs ONGC
        ]
        result = MatchResult(clusters, {}, 0.5, 4, 4)
        metrics = evaluate(result, {(0, 1), (0, 3)}, frame)
        assert metrics["cross_cpse_actual"] == 1
        assert metrics["within_cpse_actual"] == 1
        assert metrics["cross_cpse_recall"] == 1.0
        assert metrics["within_cpse_recall"] == 1.0
