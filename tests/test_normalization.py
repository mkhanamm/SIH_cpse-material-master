"""Unit tests for src.normalization -- abbreviation expansion, unit canonicalization, idempotence.

Every fixture string below is a real `Raw Description` value from the supplied
dataset, not an invented example. Tests that pass on made-up input tell you
nothing about whether the pipeline works on the data it will actually see.
"""

from __future__ import annotations

import pytest

from src.normalization import (
    ABBREVIATIONS,
    canonicalize_units,
    expand_abbreviations,
    fold_case_and_punctuation,
    normalize,
)


class TestCaseAndPunctuation:
    def test_lowercases_and_collapses_whitespace(self):
        assert fold_case_and_punctuation("  PIPE   CS  40NB ") == "pipe cs 40nb"

    def test_preserves_decimals(self):
        """1.5 must survive; a naive period-strip turns it into '1 5'."""
        assert "1.5" in fold_case_and_punctuation("CS SMLS PIPE 1.5 INCH NB")

    def test_strips_non_decimal_periods(self):
        """'GR.6' must become 'gr 6' so 'gr' can be recognised as an abbreviation."""
        assert fold_case_and_punctuation("A333 GR.6") == "a333 gr 6"

    def test_dotted_abbreviation_resolved_before_periods_are_stripped(self):
        """M.S. is unrecoverable once its periods become spaces -- regression test."""
        assert "mild steel" in fold_case_and_punctuation("M.S. SMLS PIPE DN40")

    def test_handles_non_string_input(self):
        assert fold_case_and_punctuation(None) == ""


class TestAbbreviationExpansion:
    @pytest.mark.parametrize(
        "raw,expected_token",
        [
            ("brg deep groove ball", "bearing"),
            ("cs smls pipe", "seamless"),
            ("cs smls pipe", "carbon steel"),
            ("bfly v/v lug 300", "butterfly"),
            ("bfly v/v lug 300", "valve"),
            ("300 nb cl-150", "nominal bore"),
            ("dn40 sch20", "schedule"),
            ("xlpe cable", "cross linked polyethylene"),
            ("stud m12x100 b7 c/w 2 nuts", "complete with"),
            ("plate hardox 450 16 mm thk", "thickness"),
        ],
    )
    def test_known_abbreviations_expand(self, raw, expected_token):
        assert expected_token in expand_abbreviations(raw)

    def test_splits_fused_alpha_digit(self):
        assert "schedule 20" in expand_abbreviations("sch20")

    def test_splits_fused_digit_alpha(self):
        """'300MM' and '300 NB' must converge -- regression test."""
        assert "300 mm" in expand_abbreviations("300mm class 150")

    def test_does_not_expand_by_substring(self):
        """'cs' expands, but 'csx' is not carbon steel."""
        assert "carbon steel" not in expand_abbreviations("csx housing")

    def test_every_table_entry_is_lowercase(self):
        """Lookup is by folded token; an uppercase key would be dead code."""
        assert all(k == k.lower() for k in ABBREVIATIONS)


class TestUnitCanonicalization:
    def test_metre_aliases_collapse(self):
        assert canonicalize_units("20 mtr head") == "20 m head"

    def test_compound_unit_survives_alias_pass(self):
        """m3/hr must not be rewritten into 'cubic m per hr' -- regression test."""
        result = normalize("CENTRIFUGAL PUMP FLOW 50 M3/HR").text
        assert "cumperhr" in result
        assert "per hr" not in result

    def test_compound_unit_matches_without_leading_space(self):
        """'50M3/HR' has no word boundary before m3 -- regression test."""
        assert "cumperhr" in normalize("PUMP CENTRIFUGAL 50M3/HR 40 MTR HEAD").text


class TestFullPipeline:
    def test_known_duplicate_pair_converges(self):
        """The two records below are one ground-truth group in the dataset."""
        a = normalize("BFLY V/V LUG 300 NB CL-150").text
        b = normalize("BUTTERFLY VALVE LUG TYPE 300MM CLASS 150").text
        shared = set(a.split()) & set(b.split())
        assert {"butterfly", "valve", "lug", "300", "class", "150"} <= shared

    def test_bearing_pair_becomes_token_identical(self):
        a = set(normalize("BRG DEEP GROOVE BALL BORE 25 MM").text.split())
        b = set(normalize("DEEP GROOVE BALL BEARING 25MM BORE").text.split())
        assert a == b

    def test_distinct_materials_stay_distinct(self):
        """Normalization must not erase the difference between Sch 20 and Sch 40."""
        a = normalize("SEAMLESS CARBON STEEL PIPE 40 MM DIA, SCH-20").text
        b = normalize("SEAMLESS CARBON STEEL PIPE 40 MM DIA, SCH-40").text
        assert a != b

    def test_is_idempotent(self):
        """Normalizing twice must equal normalizing once, or scores drift by call count."""
        once = normalize("CS SMLS PIPE 1.5 INCH NB SCH20 A333 GR.6").text
        assert normalize(once).text == once

    def test_trace_records_every_pass(self):
        result = normalize("BFLY V/V 300 NB")
        assert [name for name, _, _ in result.trace] == [
            "case_punctuation",
            "abbreviations",
            "units",
        ]
        assert result.original == "BFLY V/V 300 NB"

    def test_empty_input_is_safe(self):
        assert normalize("").text == ""
        assert normalize(None).text == ""
