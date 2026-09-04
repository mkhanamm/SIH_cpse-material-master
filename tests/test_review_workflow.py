"""Unit tests for src.review_workflow -- decision validity, pair labelling, uncertainty ordering."""

from __future__ import annotations

import pytest

from src.review_workflow import (
    APPROVE,
    EDIT,
    REJECT,
    DecisionRecord,
    ReviewItem,
    apply_decisions,
    build_queue,
    decisions_to_labels,
    record_decision,
)


class FakeCluster:
    """Minimal stand-in for a MaterialCluster."""

    def __init__(self, cluster_id, members, tier="MEDIUM", min_score=0.6):
        self.cluster_id = cluster_id
        self.members = members
        self.tier = tier
        self.min_score = min_score
        self.mean_score = min_score
        self.cpses = ["ONGC", "BPCL"]
        self.tier_reason = "queued"
        self.edges = [
            (a, b, min_score)
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]


@pytest.fixture
def item() -> ReviewItem:
    return ReviewItem(
        item_id=0, cluster_id=7, members=[1, 2, 3], tier="MEDIUM",
        score=0.6, uncertainty=0.05,
    )


class TestQueueOrdering:
    def test_sorted_by_uncertainty_not_score(self):
        """Showing the reviewer what the model already knows teaches it nothing."""
        clusters = [
            FakeCluster(0, [0, 1], min_score=0.95),
            FakeCluster(1, [2, 3], min_score=0.61),
            FakeCluster(2, [4, 5], min_score=0.80),
        ]
        queue = build_queue(clusters, threshold=0.60)
        assert [i.cluster_id for i in queue] == [1, 2, 0]

    def test_only_requested_tiers_queued(self):
        clusters = [
            FakeCluster(0, [0, 1], tier="HIGH"),
            FakeCluster(1, [2, 3], tier="MEDIUM"),
        ]
        assert len(build_queue(clusters, 0.6, tiers=("MEDIUM",))) == 1

    def test_item_ids_are_reassigned_after_sorting(self):
        clusters = [FakeCluster(i, [i, i + 1], min_score=0.9 - i * 0.1) for i in range(3)]
        queue = build_queue(clusters, 0.6)
        assert [i.item_id for i in queue] == [0, 1, 2]

    def test_limit_respected(self):
        clusters = [FakeCluster(i, [i, i + 1]) for i in range(5)]
        assert len(build_queue(clusters, 0.6, limit=2)) == 2


class TestDecisionValidity:
    def test_unknown_action_rejected(self, item, tmp_path):
        with pytest.raises(ValueError, match="Unknown action"):
            record_decision(item, "maybe", "alice", log_path=tmp_path / "d.jsonl")

    def test_reviewer_identity_required(self, item, tmp_path):
        with pytest.raises(ValueError, match="Reviewer identity is required"):
            record_decision(item, APPROVE, "", log_path=tmp_path / "d.jsonl")

    def test_edit_without_correction_rejected(self, item, tmp_path):
        with pytest.raises(ValueError, match="must supply corrected_members"):
            record_decision(item, EDIT, "alice", log_path=tmp_path / "d.jsonl")

    def test_decision_persists(self, item, tmp_path):
        path = tmp_path / "d.jsonl"
        record_decision(item, APPROVE, "alice", log_path=path)
        assert path.exists()

    def test_original_membership_retained_on_edit(self, item, tmp_path):
        decision = record_decision(
            item, EDIT, "alice", corrected_members=[1, 2],
            log_path=tmp_path / "d.jsonl",
        )
        assert decision.members == [1, 2]
        assert decision.original_members == [1, 2, 3]


class TestPairLabelling:
    def test_approve_labels_all_internal_pairs_positive(self):
        decision = DecisionRecord(0, 7, APPROVE, "a", "t", members=[1, 2, 3],
                                  original_members=[1, 2, 3])
        assert sorted(decision.labelled_pairs()) == [(1, 2, 1), (1, 3, 1), (2, 3, 1)]

    def test_reject_labels_all_internal_pairs_negative(self):
        decision = DecisionRecord(0, 7, REJECT, "a", "t", members=[1, 2],
                                  original_members=[1, 2])
        assert decision.labelled_pairs() == [(1, 2, 0)]

    def test_edit_labels_kept_positive_and_removed_negative(self):
        """An edit carries more information than a bare reject -- this is why."""
        decision = DecisionRecord(0, 7, EDIT, "a", "t", members=[1, 2],
                                  original_members=[1, 2, 3])
        labels = dict(((a, b), v) for a, b, v in decision.labelled_pairs())
        assert labels[(1, 2)] == 1
        assert labels[(1, 3)] == 0
        assert labels[(2, 3)] == 0

    def test_later_decisions_override_earlier(self):
        first = DecisionRecord(0, 7, APPROVE, "a", "t1", members=[1, 2],
                               original_members=[1, 2])
        second = DecisionRecord(1, 7, REJECT, "b", "t2", members=[1, 2],
                                original_members=[1, 2])
        assert decisions_to_labels([first, second])[(1, 2)] == 0


class TestApplyDecisions:
    def test_rejected_cluster_is_dropped(self):
        clusters = [FakeCluster(7, [1, 2])]
        decision = DecisionRecord(0, 7, REJECT, "a", "t", members=[1, 2],
                                  original_members=[1, 2])
        assert apply_decisions(clusters, [decision]) == []

    def test_approved_cluster_is_promoted(self):
        clusters = [FakeCluster(7, [1, 2])]
        decision = DecisionRecord(0, 7, APPROVE, "a", "t", members=[1, 2],
                                  original_members=[1, 2])
        applied = apply_decisions(clusters, [decision])
        assert applied[0].tier == "HIGH"
        assert "a" in applied[0].tier_reason

    def test_edited_cluster_is_trimmed(self):
        clusters = [FakeCluster(7, [1, 2, 3])]
        decision = DecisionRecord(0, 7, EDIT, "a", "t", members=[1, 2],
                                  original_members=[1, 2, 3])
        applied = apply_decisions(clusters, [decision])
        assert applied[0].members == [1, 2]

    def test_edit_leaving_one_record_drops_the_cluster(self):
        clusters = [FakeCluster(7, [1, 2, 3])]
        decision = DecisionRecord(0, 7, EDIT, "a", "t", members=[1],
                                  original_members=[1, 2, 3])
        assert apply_decisions(clusters, [decision]) == []

    def test_undecided_clusters_pass_through_unchanged(self):
        clusters = [FakeCluster(7, [1, 2]), FakeCluster(8, [3, 4])]
        decision = DecisionRecord(0, 7, APPROVE, "a", "t", members=[1, 2],
                                  original_members=[1, 2])
        applied = apply_decisions(clusters, [decision])
        assert len(applied) == 2
        assert applied[1].tier == "MEDIUM"

    def test_original_clusters_are_not_mutated(self):
        clusters = [FakeCluster(7, [1, 2, 3])]
        decision = DecisionRecord(0, 7, EDIT, "a", "t", members=[1, 2],
                                  original_members=[1, 2, 3])
        apply_decisions(clusters, [decision])
        assert clusters[0].members == [1, 2, 3]
