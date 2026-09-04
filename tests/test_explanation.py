"""Unit tests for src.explanation -- evidence selection, determinism, serialisability."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.explanation import (
    AGAINST,
    MISSING,
    SUPPORT,
    Explanation,
    explain_pair,
    format_value,
    label_for,
    render_explanation,
)
from src.similarity import AGREE, CONFLICT, UNKNOWN


@pytest.fixture
def pair_row() -> pd.Series:
    return pd.Series(
        {
            "idx_a": 0,
            "idx_b": 1,
            "semantic": 0.88,
            "string": 0.94,
            "attribute": 0.67,
            "match_probability": 0.91,
            "n_comparable_attrs": 3,
            "flags": {
                "nominal_size_mm": AGREE,
                "grade": AGREE,
                "schedule": CONFLICT,
                "spec_standard": UNKNOWN,
                "head_m": UNKNOWN,
            },
        }
    )


ATTRS_A = {
    "nominal_size_mm": 40.0, "grade": "astm a106", "schedule": "20",
    "spec_standard": "astm a106", "head_m": None,
}
ATTRS_B = {
    "nominal_size_mm": 40.0, "grade": "astm a106", "schedule": "40",
    "spec_standard": None, "head_m": None,
}


class TestFormatting:
    def test_numeric_value_carries_its_unit(self):
        assert format_value("nominal_size_mm", 40.0) == "40 mm"

    def test_missing_value_is_named_not_zeroed(self):
        assert format_value("nominal_size_mm", None) == "not stated"

    def test_nan_is_missing(self):
        assert format_value("nominal_size_mm", float("nan")) == "not stated"

    def test_label_is_human_readable(self):
        assert label_for("nominal_size_mm") == "Nominal bore"

    def test_unknown_attribute_still_gets_a_label(self):
        assert label_for("some_new_field") == "Some new field"


class TestPairExplanation:
    def test_agreeing_attributes_become_supporting_evidence(self, pair_row):
        explanation = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH")
        supporting = " ".join(line.text for line in explanation.supporting())
        assert "Nominal bore" in supporting
        assert "Material grade" in supporting

    def test_conflict_becomes_opposing_evidence(self, pair_row):
        explanation = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH")
        assert any("Schedule differs" in line.text for line in explanation.opposing())

    def test_hard_conflict_is_marked_procurement_critical(self, pair_row):
        explanation = explain_pair(
            pair_row, ATTRS_A, ATTRS_B, "HIGH",
            hard_conflict_attributes=("schedule",),
        )
        assert any(
            "procurement-critical" in line.text for line in explanation.opposing()
        )

    def test_one_sided_attribute_is_flagged(self, pair_row):
        """Record A states a spec, B does not -- a reviewer can act on that."""
        explanation = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH")
        assert any(
            line.marker == MISSING and "Specification" in line.text
            for line in explanation.lines
        )

    def test_attribute_absent_from_both_produces_no_line(self, pair_row):
        """Otherwise real evidence is buried under a dozen 'not stated' rows."""
        explanation = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH")
        assert not any("Head" in line.text for line in explanation.lines)

    def test_channel_scores_are_shown(self, pair_row):
        explanation = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH")
        assert explanation.channel_scores["semantic"] == pytest.approx(0.88)
        assert any("semantic 0.88" in line.text for line in explanation.lines)

    def test_decisive_evidence_is_ordered_first(self, pair_row):
        explanation = explain_pair(
            pair_row, ATTRS_A, ATTRS_B, "HIGH",
            hard_conflict_attributes=("schedule",),
        )
        assert explanation.lines[0].marker in (SUPPORT, AGAINST)

    def test_no_comparable_attributes_raises_a_caveat(self):
        row = pd.Series(
            {
                "idx_a": 0, "idx_b": 1, "semantic": 0.9, "string": 0.9,
                "attribute": 0.5, "match_probability": 0.8,
                "n_comparable_attrs": 0, "flags": {},
            }
        )
        assert explain_pair(row, {}, {}, "UNKNOWN").caveats


class TestDeterminismAndSerialisation:
    def test_output_is_byte_identical_across_calls(self, pair_row):
        """The reason a template is used instead of a live LLM in the demo."""
        first = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH").as_text()
        second = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH").as_text()
        assert first == second

    def test_dict_form_is_json_serialisable(self, pair_row):
        """The audit trail stores this; it must survive a round-trip to disk."""
        payload = explain_pair(pair_row, ATTRS_A, ATTRS_B, "HIGH").as_dict()
        assert json.loads(json.dumps(payload))["verdict"] == "HIGH"

    def test_rendering_includes_verdict_and_score(self):
        explanation = Explanation(verdict="HIGH", score=0.91, subject="pair 0-1")
        rendered = render_explanation(explanation)
        assert "HIGH" in rendered and "0.91" in rendered

    def test_caveats_are_rendered(self):
        explanation = Explanation(
            verdict="UNKNOWN", score=0.4, caveats=["Insufficient data."]
        )
        assert "Insufficient data." in render_explanation(explanation)
