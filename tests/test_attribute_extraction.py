"""Unit tests for src.attribute_extraction -- regex correctness, graceful degradation on nulls, no fabricated values.

Fixtures are real `Raw Description` values from the supplied dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.attribute_extraction import (
    INCH_TO_NB_MM,
    AttributeSet,
    _values_agree,
    extract,
    inch_to_nb_mm,
)


class TestNominalBoreConversion:
    def test_inch_nb_uses_designation_not_arithmetic(self):
        """1.5in NB is designated 40mm, not 38.1mm. This is the whole point."""
        assert inch_to_nb_mm(1.5) == 40.0
        assert inch_to_nb_mm(1.5) != pytest.approx(1.5 * 25.4, rel=0.01)

    @pytest.mark.parametrize(
        "inches,expected", [(0.5, 15.0), (2.0, 50.0), (6.0, 150.0), (12.0, 300.0)]
    )
    def test_standard_sizes(self, inches, expected):
        assert inch_to_nb_mm(inches) == expected

    def test_non_standard_size_returns_none_rather_than_guessing(self):
        assert inch_to_nb_mm(7.3) is None

    def test_table_is_monotonic(self):
        sizes = [INCH_TO_NB_MM[k] for k in sorted(INCH_TO_NB_MM)]
        assert sizes == sorted(sizes)

    def test_inch_and_mm_records_converge(self):
        """These two are one ground-truth group in the dataset."""
        a = extract("CS SMLS PIPE 1.5 INCH NB SCH20 A333 GR.6")
        b = extract("M.S. SMLS PIPE DN40 SCH20")
        assert a.nominal_size_mm == b.nominal_size_mm == 40.0


class TestExtraction:
    def test_pipe_attributes(self):
        attrs = extract("PIPE CS 40NB SCH 20 A106-B")
        assert attrs.nominal_size_mm == 40.0
        assert attrs.schedule == "20"
        assert attrs.material_of_construction == "carbon steel"

    def test_schedule_distinguishes_near_identical_records(self):
        """Sch 20 vs Sch 40 is the discriminator the text channel cannot see."""
        a = extract("SEAMLESS CARBON STEEL PIPE 40 MM DIA, SCH-20, ASTM A106")
        b = extract("SEAMLESS CARBON STEEL PIPE 40 MM DIA, SCH-40, ASTM A106")
        assert a.schedule == "20" and b.schedule == "40"
        assert a.nominal_size_mm == b.nominal_size_mm

    def test_mild_steel_maps_to_carbon_steel_class(self):
        """MS and CS are one procurement class; treating them as different splits groups."""
        assert extract("M.S. SMLS PIPE DN40").material_of_construction == "carbon steel"

    def test_pressure_class_and_size(self):
        attrs = extract("BFLY V/V LUG 300 NB CL-150")
        assert attrs.nominal_size_mm == 300.0
        assert attrs.pressure_class == "150"

    def test_flow_and_head_are_separate_attributes(self):
        attrs = extract("CENTRIFUGAL PUMP FLOW 50 M3/HR HEAD 100M")
        assert attrs.flow_m3hr == 50.0
        assert attrs.head_m == 100.0

    def test_flow_rate_is_not_read_as_a_volume_capacity(self):
        """'cumperhr' must not satisfy the 'cum' capacity pattern -- regression test."""
        assert extract("PUMP CENTRIFUGAL 50M3/HR 40 MTR HEAD").capacity_cum is None

    def test_transformer_rating(self):
        attrs = extract("DISTRIBUTION TRANSFORMER 63KVA 11/0.433 KV")
        assert attrs.rating_kva == 63.0

    def test_gearbox_ratio(self):
        assert extract("HELICAL GEARBOX 5HP RATIO 15:1 FOOT MTD").ratio == "15:1"


class TestBareMillimetreFallback:
    """A bare '100mm' with no NB/bore/inch cue -- the only size source on a
    real CPSE upload where just the four required columns are mapped."""

    def test_gate_valve_bare_mm_is_extracted(self):
        attrs = extract("GATE VALVE 100MM CLASS 150 CARBON STEEL BODY")
        assert attrs.nominal_size_mm == 100.0

    def test_bare_mm_provenance_is_marked_weak_not_confident(self):
        attrs = extract("GATE VALVE 100MM CLASS 150 CARBON STEEL BODY")
        assert attrs.provenance["nominal_size_mm"] == "regex_weak"

    def test_confident_nb_match_keeps_plain_regex_provenance(self):
        attrs = extract("GATE VALVE 100 NB CLASS 150")
        assert attrs.nominal_size_mm == 100.0
        assert attrs.provenance["nominal_size_mm"] == "regex"

    def test_bare_and_nb_records_converge_on_the_same_size(self):
        bare = extract("CAST IRON GATE VALVE 100MM PN16")
        nb = extract("GATE VALVE 100 NB CLASS 150 CS")
        assert bare.nominal_size_mm == nb.nominal_size_mm == 100.0

    def test_thickness_is_not_read_as_a_nominal_size(self):
        attrs = extract("MS PLATE 3000 LONG 6 MM THICKNESS")
        assert attrs.thickness_mm == 6.0
        assert attrs.nominal_size_mm is None

    def test_width_is_not_read_as_a_nominal_size(self):
        attrs = extract("CONVEYOR BELT 800 MM WIDTH 3 PLY")
        assert attrs.width_mm == 800.0
        assert attrs.nominal_size_mm is None

    def test_dimension_pair_tail_is_not_read_as_a_nominal_size(self):
        """'3000 x 1500 mm' is a plate size -- the 1500 must not become a bore."""
        attrs = extract("CARBON STEEL PLATE 3000 X 1500 MM")
        assert attrs.nominal_size_mm is None

    def test_bore_keyword_still_takes_the_confident_path(self):
        attrs = extract("DEEP GROOVE BALL BEARING 25 MM BORE")
        assert attrs.nominal_size_mm == 25.0
        assert attrs.provenance["nominal_size_mm"] == "regex"

    def test_no_measurement_still_degrades_to_none(self):
        assert extract("APRON FEEDER PAN").nominal_size_mm is None


class TestGracefulDegradation:
    def test_absent_attributes_are_none_not_defaults(self):
        """A fabricated zero would read downstream as a real, matching value."""
        attrs = extract("APRON FEEDER PAN")
        assert attrs.nominal_size_mm is None
        assert attrs.schedule is None
        assert attrs.n_known() == 0

    def test_provenance_marks_unknowns(self):
        attrs = extract("APRON FEEDER PAN")
        assert attrs.provenance["schedule"] == "unknown"

    def test_known_excludes_nones(self):
        attrs = extract("PIPE CS 40NB SCH 20")
        assert all(v is not None for v in attrs.known().values())
        assert "provenance" not in attrs.known()

    def test_null_attribute_columns_do_not_crash(self):
        row = pd.Series(
            {
                "Raw Description": "PIPE CS 40NB SCH 20",
                "Material/Grade": None,
                "Dimensions": float("nan"),
                "Specification/Standard": None,
                "Capacity/Rating": None,
                "Operating Parameter": None,
            }
        )
        assert extract(row["Raw Description"], row).nominal_size_mm == 40.0

    def test_column_and_text_agreement_is_recorded(self):
        row = pd.Series(
            {
                "Raw Description": "PIPE CS 40NB SCH 20 A106-B",
                "Material/Grade": "ASTM A106 Grade B",
                "Dimensions": "40 mm NB, Schedule 20",
                "Specification/Standard": "ASTM A106",
                "Capacity/Rating": None,
                "Operating Parameter": None,
            }
        )
        attrs = extract(row["Raw Description"], row)
        assert attrs.provenance["nominal_size_mm"] in {"both", "column"}
        assert attrs.nominal_size_mm == 40.0

    def test_empty_description(self):
        assert extract("").n_known() == 0


class TestValueAgreement:
    def test_numeric_within_tolerance(self):
        assert _values_agree(40.0, 40.5) is True

    def test_numeric_outside_tolerance(self):
        assert _values_agree(40.0, 50.0) is False

    def test_string_containment(self):
        assert _values_agree("astm a106", "astm a106 grade b") is True

    def test_different_specs_disagree(self):
        assert _values_agree("astm a106", "astm a333") is False

    def test_zero_is_not_treated_as_close_to_everything(self):
        assert _values_agree(0.0, 5.0) is False


class TestAttributeSetContract:
    def test_n_known_counts_only_populated(self):
        assert AttributeSet(nominal_size_mm=40.0, schedule="20").n_known() == 2

    def test_default_is_entirely_unknown(self):
        assert AttributeSet().n_known() == 0
