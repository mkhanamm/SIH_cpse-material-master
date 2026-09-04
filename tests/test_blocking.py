"""Unit tests for src.blocking -- key construction, multi-key emission, honest recall accounting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.blocking import (
    GENERIC_CATEGORY_TOKENS,
    _size_bucket,
    _spec_family,
    attribute_signature_key,
    candidate_pairs,
    category_key,
    category_token_key,
    evaluate_blocking,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Records reproducing the taxonomy disagreement seen in the real data."""
    return pd.DataFrame(
        {
            "Material Category": [
                "Conveyor Component",
                "Conveyor Idler",
                "Globe Valve",
                "Valve",
                "Seamless Pipe",
            ],
            "nominal_size_mm": [108.0, 108.0, 50.0, 50.0, 40.0],
            "spec_standard": [None, None, None, None, "astm a106"],
            "material_of_construction": [None, None, None, None, "carbon steel"],
            "capacity_cum": [None] * 5,
            "rating_kva": [None] * 5,
            "CPSE": ["CIL", "NMDC", "ONGC", "BPCL", "HPCL"],
        }
    )


class TestSizeBucketing:
    def test_none_is_not_a_bucket_value(self):
        assert _size_bucket(None) == "na"

    def test_nan_is_not_a_bucket_value(self):
        assert _size_bucket(float("nan")) == "na"

    def test_nearby_sizes_share_a_bucket(self):
        assert _size_bucket(40.0) == _size_bucket(45.0)

    def test_distant_sizes_do_not(self):
        assert _size_bucket(40.0) != _size_bucket(300.0)


class TestSpecFamily:
    def test_grade_suffix_folds_into_family(self):
        assert _spec_family("astm a106 grade b") == _spec_family("astm a106")

    def test_different_bodies_differ(self):
        assert _spec_family("astm a106") != _spec_family("is 2062")

    def test_missing_spec(self):
        assert _spec_family(None) == "na"


class TestCategoryKeys:
    def test_same_category_and_size_collide(self, frame):
        rows = frame.loc[[2]], frame.loc[[3]]
        assert category_key(frame.loc[2]) != category_key(frame.loc[3])

    def test_token_key_bridges_globe_valve_and_valve(self, frame):
        """The real dataset loses 15 true pairs to exactly this disagreement."""
        shared = set(category_token_key(frame.loc[2])) & set(
            category_token_key(frame.loc[3])
        )
        assert shared

    def test_token_key_bridges_conveyor_categories(self, frame):
        shared = set(category_token_key(frame.loc[0])) & set(
            category_token_key(frame.loc[1])
        )
        assert shared

    def test_token_key_does_not_bridge_unrelated_categories(self, frame):
        assert not (
            set(category_token_key(frame.loc[0])) & set(category_token_key(frame.loc[4]))
        )

    def test_generic_tokens_excluded(self, frame):
        keys = category_token_key(frame.loc[0])
        assert all("component" not in k for k in keys)
        assert "component" in GENERIC_CATEGORY_TOKENS

    def test_token_key_returns_a_list(self, frame):
        assert isinstance(category_token_key(frame.loc[0]), list)


class TestAttributeSignature:
    def test_requires_a_known_size(self, frame):
        row = frame.loc[0].copy()
        row["nominal_size_mm"] = None
        assert attribute_signature_key(row) is None

    def test_requires_more_than_size_alone(self, frame):
        """Size-only signatures bucket thousands of unrelated records together."""
        assert attribute_signature_key(frame.loc[0]) is None

    def test_populated_record_gets_a_key(self, frame):
        assert attribute_signature_key(frame.loc[4]) is not None


class TestCandidatePairs:
    def test_pairs_are_ordered_and_deduplicated(self, frame):
        result = candidate_pairs(frame, ["category_size", "category_token"])
        assert all(a < b for a, b in result.pairs)
        assert len(result.pairs) == len(set(result.pairs))

    def test_pair_sources_record_which_family_proposed_each(self, frame):
        result = candidate_pairs(frame, ["category_size", "category_token"])
        assert all(result.pair_sources[p] for p in result.pairs)

    def test_reduction_is_reported(self, frame):
        result = candidate_pairs(frame, ["category_size", "category_token"])
        assert result.stats.naive_comparisons == 10  # 5 records
        assert result.stats.blocked_comparisons <= 10

    def test_unrelated_records_are_not_paired(self, frame):
        result = candidate_pairs(frame, ["category_size", "category_token"])
        assert (0, 4) not in result.pairs


class TestBlockingEvaluation:
    def test_recall_ceiling_charged_for_lost_pairs(self, frame):
        result = candidate_pairs(frame, ["category_size"])
        # (0, 1) spans two categories, so exact-category blocking cannot see it.
        stats = evaluate_blocking(result, {(0, 1)})
        assert stats.pairs_lost == 1
        assert stats.recall_ceiling == 0.0

    def test_token_key_recovers_that_pair(self, frame):
        result = candidate_pairs(frame, ["category_size", "category_token"])
        stats = evaluate_blocking(result, {(0, 1)})
        assert stats.pairs_lost == 0
        assert stats.recall_ceiling == 1.0

    def test_summary_line_reports_both_cost_and_recall(self, frame):
        result = candidate_pairs(frame, ["category_size", "category_token"])
        line = evaluate_blocking(result, {(0, 1)}).summary_line()
        assert "reduction" in line and "recall ceiling" in line
