"""Unit tests for src.classifier -- leakage-free splitting, threshold tuning, tier calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.classifier import (
    FEATURE_COLUMNS,
    build_features,
    calibrate_tiers,
    fallback_tiers,
    group_disjoint_split,
    is_backend_mismatch,
    label_pairs,
    load_model,
    load_model_backend,
    load_model_or_none,
    pretrained_tiers,
    save_model,
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


class TestFallbackTiers:
    """Tier cut-offs used when a dataset has no GroundTruth_Group to calibrate from."""

    def test_defaults_to_config_fused_thresholds(self):
        tiers = fallback_tiers()
        assert tiers.high == config.FUSED_HIGH_CONFIDENCE_THRESHOLD
        assert tiers.medium == config.FUSED_MEDIUM_CONFIDENCE_THRESHOLD

    def test_marked_as_not_calibrated(self):
        assert fallback_tiers().calibrated is False

    def test_calibrated_tiers_default_to_true(self):
        """A normally-calibrated TierThresholds must not look like a fallback."""
        y = np.array([0, 0, 1, 1])
        tiers = calibrate_tiers(y, np.array([0.1, 0.2, 0.8, 0.9]))
        assert tiers.calibrated is True

    def test_custom_thresholds_are_honoured(self):
        tiers = fallback_tiers(high=0.9, medium=0.6)
        assert (tiers.high, tiers.medium) == (0.9, 0.6)

    def test_summary_lines_do_not_mention_validation(self):
        """Nothing was calibrated, so the summary must not claim it was."""
        lines = "\n".join(fallback_tiers().summary_lines())
        assert "validation" not in lines.lower()
        assert "fixed fallback" in lines.lower()

    def test_summary_lines_render_without_error(self):
        assert len(fallback_tiers().summary_lines()) == 3


class TestPretrainedTiers:
    """Tier cut-offs for the shipped classifier when there is no ground truth."""

    def test_defaults_to_config_pretrained_thresholds(self):
        tiers = pretrained_tiers()
        assert tiers.high == config.PRETRAINED_HIGH_THRESHOLD
        assert tiers.medium == config.PRETRAINED_EDGE_THRESHOLD

    def test_marked_as_not_calibrated(self):
        """Calibrated on the demo dataset, not on whatever is currently loaded."""
        assert pretrained_tiers().calibrated is False

    def test_custom_thresholds_are_honoured(self):
        tiers = pretrained_tiers(high=0.9, medium=0.6)
        assert (tiers.high, tiers.medium) == (0.9, 0.6)

    def test_summary_lines_render_without_error(self):
        assert len(pretrained_tiers().summary_lines()) == 3

    def test_matches_the_calibrated_values_from_the_demo_dataset(self):
        assert config.PRETRAINED_EDGE_THRESHOLD == 0.55
        assert config.PRETRAINED_HIGH_THRESHOLD == 0.85


class TestLoadModelOrNone:
    def test_returns_none_when_missing(self, tmp_path):
        assert load_model_or_none(tmp_path / "does_not_exist.pkl") is None

    def test_returns_pipeline_when_present(self, tmp_path):
        """Old, bare-pipeline artifact format -- no backend tag."""
        path = tmp_path / "classifier.pkl"
        save_model("a fitted pipeline stand-in", path)
        assert load_model_or_none(path) == "a fitted pipeline stand-in"

    def test_unwraps_dict_format_artifact(self, tmp_path):
        """New format tags the backend; load_model_or_none must still hand
        back the bare pipeline, not the wrapper dict."""
        path = tmp_path / "classifier.pkl"
        save_model("a fitted pipeline stand-in", path, backend="tfidf_svd")
        assert load_model_or_none(path) == "a fitted pipeline stand-in"


class TestLoadModelBackend:
    def test_missing_artifact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_backend(tmp_path / "does_not_exist.pkl")

    def test_returns_recorded_backend(self, tmp_path):
        path = tmp_path / "classifier.pkl"
        save_model("a fitted pipeline stand-in", path, backend="tfidf_svd")
        assert load_model_backend(path) == "tfidf_svd"

    def test_old_bare_pipeline_format_reports_unknown(self, tmp_path):
        """A pre-existing artifact from before the backend tag existed."""
        path = tmp_path / "classifier.pkl"
        save_model("a fitted pipeline stand-in", path)
        assert load_model_backend(path) is None

    def test_load_model_unwraps_dict_format(self, tmp_path):
        """load_model() stays backward compatible: same pipeline either way."""
        path = tmp_path / "classifier.pkl"
        save_model("a fitted pipeline stand-in", path, backend="sbert")
        assert load_model(path) == "a fitted pipeline stand-in"


class TestBackendMismatchDetection:
    """The bug this guards against: config.SEMANTIC_BACKEND='auto' can resolve
    to a different backend on the machine that trained models/classifier.pkl
    than on the machine running it, silently miscalibrating match_probability."""

    def test_matching_backends_is_not_a_mismatch(self):
        assert is_backend_mismatch("tfidf_svd", "tfidf_svd") is False

    def test_differing_backends_is_a_mismatch(self):
        assert is_backend_mismatch("tfidf_svd", "sbert") is True
        assert is_backend_mismatch("sbert", "tfidf_svd") is True

    def test_unknown_trained_backend_is_not_reported_as_mismatch(self):
        """An old bare-pipeline artifact recorded nothing to compare against;
        that must not raise a false alarm on every load."""
        assert is_backend_mismatch(None, "sbert") is False
        assert is_backend_mismatch(None, "tfidf_svd") is False

    def test_end_to_end_via_saved_artifact(self, tmp_path):
        """Round-trip through save_model/load_model_backend into the check."""
        path = tmp_path / "classifier.pkl"
        save_model("a fitted pipeline stand-in", path, backend="tfidf_svd")
        trained = load_model_backend(path)
        assert is_backend_mismatch(trained, "sbert") is True
        assert is_backend_mismatch(trained, "tfidf_svd") is False
