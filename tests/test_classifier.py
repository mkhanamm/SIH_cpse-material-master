"""Unit tests for src.classifier -- leakage-free splitting, threshold tuning, tier calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.classifier import (
    FEATURE_COLUMNS,
    build_features,
    calibrate_tiers,
    group_disjoint_split,
    label_pairs,
    tune_threshold,
)


@pytest.fixture
def eval_df() -> pd.DataFrame:
    """Six records in three ground-truth groups of two."""
    return pd.DataFrame(
        {"GroundTruth_Group": ["G1", "G1", "G2", "G2", "G3", "G3"]}
    )


@pytest.fixture
def scored_pairs() -> pd.DataFrame:
    pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 4)]
    return pd.DataFrame(
        {
            "idx_a": [a for a, _ in pairs],
            "idx_b": [b for _, b in pairs],
            **{c: [0.5] * len(pairs) for c in FEATURE_COLUMNS},
            "cross_cpse": [True] * len(pairs),
        }
    )


class TestLabelling:
    def test_same_group_is_positive(self, scored_pairs, eval_df):
        labels = label_pairs(scored_pairs, eval_df)
        assert list(labels[:3]) == [1, 1, 1]

    def test_different_groups_are_negative(self, scored_pairs, eval_df):
        labels = label_pairs(scored_pairs, eval_df)
        assert list(labels[3:]) == [0, 0]


class TestFeatures:
    def test_fused_score_excluded(self):
        """Including it would smuggle the hand-tuned weights into the model."""
        assert "fused" not in FEATURE_COLUMNS

    def test_shape_matches_pairs_and_features(self, scored_pairs):
        X, names = build_features(scored_pairs)
        assert X.shape == (len(scored_pairs), len(FEATURE_COLUMNS))
        assert names == FEATURE_COLUMNS


class TestGroupDisjointSplit:
    def test_splits_are_mutually_exclusive(self, scored_pairs, eval_df):
        train, val, test, _ = group_disjoint_split(scored_pairs, eval_df)
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val & test).any()

    def test_no_group_appears_on_two_sides(self, scored_pairs, eval_df):
        train, val, test, _ = group_disjoint_split(scored_pairs, eval_df)
        groups = eval_df["GroundTruth_Group"]

        def groups_in(mask):
            subset = scored_pairs[mask]
            return set(groups.reindex(subset["idx_a"])) | set(
                groups.reindex(subset["idx_b"])
            )

        assert not (groups_in(train) & groups_in(test))
        assert not (groups_in(val) & groups_in(test))

    def test_straddling_pairs_are_discarded_not_assigned(self, scored_pairs, eval_df):
        train, val, test, discarded = group_disjoint_split(scored_pairs, eval_df)
        assert train.sum() + val.sum() + test.sum() + discarded == len(scored_pairs)

    def test_is_deterministic(self, scored_pairs, eval_df):
        first = group_disjoint_split(scored_pairs, eval_df)
        second = group_disjoint_split(scored_pairs, eval_df)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[2], second[2])


class TestThresholdTuning:
    def test_finds_a_separating_threshold(self):
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, f1 = tune_threshold(y, probabilities)
        assert f1 == 1.0
        assert 0.2 < threshold <= 0.8

    def test_no_positives_falls_back_safely(self):
        assert tune_threshold(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])) == (0.5, 0.0)

    def test_does_not_assume_half(self):
        """Balanced class weights move the operating point away from 0.5."""
        y = np.array([0, 0, 0, 1])
        probabilities = np.array([0.01, 0.02, 0.03, 0.06])
        threshold, _ = tune_threshold(y, probabilities)
        assert threshold < 0.5


class TestTierCalibration:
    def test_high_meets_precision_target_when_reachable(self):
        y = np.array([0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        tiers = calibrate_tiers(y, probabilities, high_precision_target=0.99)
        assert tiers.achieved_high_precision >= 0.99

    def test_high_never_below_medium(self):
        y = np.array([0, 1, 1, 0, 1])
        probabilities = np.array([0.4, 0.5, 0.6, 0.55, 0.7])
        tiers = calibrate_tiers(y, probabilities)
        assert tiers.high >= tiers.medium

    def test_unreachable_target_reports_achieved_not_target(self):
        """The target must not be silently relaxed to make the report look good."""
        y = np.array([0, 1, 0, 1])
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])
        tiers = calibrate_tiers(y, probabilities, high_precision_target=0.99)
        assert tiers.achieved_high_precision < 0.99
        assert tiers.high_precision_target == 0.99

    def test_summary_lines_render(self):
        y = np.array([0, 0, 1, 1])
        tiers = calibrate_tiers(y, np.array([0.1, 0.2, 0.8, 0.9]))
        assert len(tiers.summary_lines()) == 3
