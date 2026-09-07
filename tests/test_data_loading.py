"""Unit tests for src.data_loading -- upload reading, column mapping, runtime estimate."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src import config
from src.data_loading import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    UnsupportedFileType,
    apply_column_mapping,
    build_dataset_from_mapped,
    column_hint,
    duplicate_required_sources,
    estimate_runtime_seconds,
    guess_column_mapping,
    infer_column_mapping,
    preview_mapped_row,
    read_upload,
    suggest_column_mapping,
)


def _realistic_upload(n: int = 36) -> pd.DataFrame:
    """An upload with non-matching headers but realistically-shaped values."""
    companies = ["NTPC", "SAIL", "BHEL", "GAIL", "ONGC", "NMDC"]
    categories = [
        "Gate Valve", "Globe Valve", "Ball Bearing", "Seamless Pipe", "Gasket",
        "Centrifugal Pump", "Transformer", "Conveyor Idler", "Circuit Breaker",
    ]
    descriptions = [
        "GATE V/V 100 NB CL-150 CS BODY BOLTED BONNET",
        "GLOBE VALVE 80MM CLASS 300 FORGED STEEL BODY",
        "DEEP GROOVE BALL BEARING 6205 25MM BORE C3",
        "CS SEAMLESS PIPE 40 NB SCH 40 ASTM A106 GR B",
        "SPIRAL WOUND GASKET 150 NB CLASS 300 SS316 CG",
        "CENTRIFUGAL PUMP 50 M3/HR 30 M HEAD 15 KW MOTOR",
        "DISTRIBUTION TRANSFORMER 63 KVA 11/0.433 KV ONAN",
        "TROUGHING IDLER ROLLER 152 MM DIA 3 ROLL CARRY SET",
        "VACUUM CIRCUIT BREAKER 12 KV 1250 A 25 KA PANEL",
    ]
    return pd.DataFrame(
        {
            "Owner Org": [companies[i % len(companies)] for i in range(n)],
            "Part No": [
                f"{companies[i % len(companies)]}-{1000 + i}" for i in range(n)
            ],
            "Item Class": [categories[i % len(categories)] for i in range(n)],
            "Long Text": [descriptions[i % len(descriptions)] for i in range(n)],
            "UoM": ["NOS"] * n,
        }
    )


def _uploaded_frame() -> pd.DataFrame:
    """A file with headers that don't match the canonical schema."""
    return pd.DataFrame(
        {
            "Enterprise": ["ONGC", "BPCL"],
            "Material Code": ["M1", "M2"],
            "Category": ["Valve", "Valve"],
            "Description": ["BFLY V/V 300 CI", "BUTTERFLY VALVE 300 CAST IRON"],
            "Extra Column Nobody Asked For": ["x", "y"],
        }
    )


class TestReadUpload:
    def test_reads_csv(self, tmp_path):
        path = tmp_path / "upload.csv"
        _uploaded_frame().to_csv(path, index=False)
        with open(path, "rb") as handle:
            df = read_upload(handle, "upload.csv")
        assert list(df.columns) == list(_uploaded_frame().columns)
        assert len(df) == 2

    def test_reads_xlsx(self, tmp_path):
        path = tmp_path / "upload.xlsx"
        _uploaded_frame().to_excel(path, index=False, engine="openpyxl")
        with open(path, "rb") as handle:
            df = read_upload(handle, "upload.xlsx")
        assert len(df) == 2

    def test_case_insensitive_extension(self, tmp_path):
        path = tmp_path / "upload.CSV"
        _uploaded_frame().to_csv(path, index=False)
        with open(path, "rb") as handle:
            df = read_upload(handle, "upload.CSV")
        assert len(df) == 2

    def test_rejects_unsupported_extension(self):
        with pytest.raises(UnsupportedFileType, match=r"\.txt"):
            read_upload(io.BytesIO(b"not a spreadsheet"), "notes.txt")


class TestSuggestColumnMapping:
    def test_matches_exact_headers(self):
        columns = ["CPSE", "CPSE Material Code", "Material Category", "Raw Description"]
        suggested = suggest_column_mapping(columns)
        for target in REQUIRED_COLUMNS:
            assert suggested[target] == target

    def test_matches_case_and_whitespace_insensitively(self):
        suggested = suggest_column_mapping(["  cpse  ", "material category"])
        assert suggested["CPSE"] == "  cpse  "
        assert suggested["Material Category"] == "material category"

    def test_unmatched_target_is_none(self):
        suggested = suggest_column_mapping(["Enterprise", "Material Code"])
        assert suggested["CPSE"] is None
        assert suggested["Raw Description"] is None

    def test_covers_every_required_and_optional_column(self):
        suggested = suggest_column_mapping([])
        assert set(suggested) == set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)


class TestApplyColumnMapping:
    def test_renames_mapped_columns(self):
        raw = _uploaded_frame()
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        out = apply_column_mapping(raw, mapping)
        assert list(out["CPSE"]) == ["ONGC", "BPCL"]
        assert list(out["Raw Description"]) == list(raw["Description"])

    def test_unmapped_optional_columns_are_blank_not_missing(self):
        raw = _uploaded_frame()
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        out = apply_column_mapping(raw, mapping)
        assert "Sector" in out.columns
        assert (out["Sector"] == "").all()

    def test_auto_defaulted_columns_are_present(self):
        raw = _uploaded_frame()
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        out = apply_column_mapping(raw, mapping)
        assert "UOM" in out.columns
        assert "Legacy_Sector_Code" in out.columns

    def test_missing_required_column_raises(self):
        raw = _uploaded_frame()
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": None,
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        with pytest.raises(ValueError, match="CPSE Material Code"):
            apply_column_mapping(raw, mapping)

    def test_same_column_mapped_to_two_required_fields_raises(self):
        raw = _uploaded_frame()
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Description",
            "Raw Description": "Description",
        }
        with pytest.raises(ValueError, match=r"Description.*Material Category|Material Category.*Raw Description"):
            apply_column_mapping(raw, mapping)

    def test_output_is_reindexed_from_zero(self):
        raw = _uploaded_frame()
        raw.index = [10, 20]
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        out = apply_column_mapping(raw, mapping)
        assert list(out.index) == [0, 1]


class TestDuplicateRequiredSources:
    def test_one_to_one_mapping_has_no_collisions(self):
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        assert duplicate_required_sources(mapping) == {}

    def test_names_the_duplicated_column_and_the_colliding_fields(self):
        mapping = {
            "CPSE": "col_a",
            "CPSE Material Code": "col_a",
            "Material Category": "col_a",
            "Raw Description": "col_a",
        }
        collisions = duplicate_required_sources(mapping)
        assert set(collisions) == {"col_a"}
        assert collisions["col_a"] == REQUIRED_COLUMNS

    def test_partial_collision_reports_only_the_offending_pair(self):
        mapping = {
            "CPSE": "who",
            "CPSE Material Code": "code",
            "Material Category": "desc",
            "Raw Description": "desc",
        }
        collisions = duplicate_required_sources(mapping)
        assert collisions == {"desc": ["Material Category", "Raw Description"]}

    def test_optional_columns_sharing_a_source_are_not_flagged(self):
        mapping = {
            "CPSE": "who",
            "CPSE Material Code": "code",
            "Material Category": "cat",
            "Raw Description": "desc",
            "Dimensions": "notes",
            "Specification/Standard": "notes",
        }
        assert duplicate_required_sources(mapping) == {}

    def test_unmapped_required_fields_do_not_count_as_a_collision(self):
        mapping = {
            "CPSE": None,
            "CPSE Material Code": None,
            "Material Category": "cat",
            "Raw Description": "desc",
        }
        assert duplicate_required_sources(mapping) == {}


class TestColumnHint:
    def test_shows_distinct_example_values(self):
        hint = column_hint(pd.Series(["NTPC", "SAIL", "BHEL", "NTPC", "SAIL"]))
        assert "NTPC" in hint and "SAIL" in hint and "BHEL" in hint

    def test_trailing_ellipsis_when_more_values_exist(self):
        assert column_hint(pd.Series(["a", "b", "c", "d"])).endswith("...")

    def test_no_trailing_ellipsis_when_all_values_shown(self):
        assert column_hint(pd.Series(["only", "two"])) == "only, two"

    def test_long_values_are_truncated(self):
        hint = column_hint(pd.Series(["X" * 80]))
        assert len(hint) <= 25 and "…" in hint

    def test_empty_or_all_null_column_gives_empty_string(self):
        assert column_hint(pd.Series([None, float("nan"), "  "])) == ""


class TestGuessColumnMapping:
    def test_guesses_all_four_required_fields_from_data_shape(self):
        guessed = guess_column_mapping(_realistic_upload())
        assert guessed["Raw Description"] == "Long Text"
        assert guessed["CPSE Material Code"] == "Part No"
        assert guessed["CPSE"] == "Owner Org"
        assert guessed["Material Category"] == "Item Class"

    def test_longest_text_column_is_the_description(self):
        df = pd.DataFrame(
            {
                "a": ["NTPC", "SAIL"] * 10,
                "b": ["short label"] * 20,
                "c": ["a much longer free text description of the item"] * 20,
            }
        )
        assert guess_column_mapping(df)["Raw Description"] == "c"

    def test_company_column_has_fewer_distinct_values_than_category(self):
        guessed = guess_column_mapping(_realistic_upload())
        # Owner Org (6 distinct) -> CPSE; Item Class (9 distinct) -> Category.
        assert guessed["CPSE"] != guessed["Material Category"]

    def test_unique_dashed_code_column_is_the_material_code(self):
        df = pd.DataFrame(
            {
                "org": ["NTPC", "SAIL", "BHEL"] * 10,
                "ref": [f"NTPC-VLV-{2000 + i}" for i in range(30)],
                "kind": ["Gate Valve", "Globe Valve", "Ball Bearing"] * 10,
                "text": ["a fairly long free-text material description here"] * 30,
            }
        )
        assert guess_column_mapping(df)["CPSE Material Code"] == "ref"

    def test_no_columns_yields_all_none(self):
        guessed = guess_column_mapping(pd.DataFrame({"x": [None, None]}))
        assert set(guessed.values()) == {None}

    def test_never_maps_two_required_fields_to_one_column(self):
        guessed = guess_column_mapping(_realistic_upload())
        chosen = [c for c in guessed.values() if c is not None]
        assert len(chosen) == len(set(chosen))


class TestInferColumnMapping:
    def test_exact_header_match_wins_over_data_guess(self):
        df = _realistic_upload().rename(columns={"Owner Org": "CPSE"})
        inferred = infer_column_mapping(df)
        assert inferred["CPSE"] == "CPSE"

    def test_fills_required_fields_left_blank_by_header_matching(self):
        inferred = infer_column_mapping(_realistic_upload())
        for field in REQUIRED_COLUMNS:
            assert inferred[field] is not None

    def test_covers_the_same_targets_as_suggest(self):
        df = _realistic_upload()
        assert set(infer_column_mapping(df)) == set(
            suggest_column_mapping(list(df.columns))
        )

    def test_does_not_reuse_a_column_already_claimed_by_a_header_match(self):
        # "Part No" renamed to the canonical code header; the guesser must not
        # then also point CPSE or Category at it.
        df = _realistic_upload().rename(columns={"Part No": "CPSE Material Code"})
        inferred = infer_column_mapping(df)
        claimed = [inferred[f] for f in REQUIRED_COLUMNS]
        assert len(claimed) == len(set(claimed))


class TestPreviewMappedRow:
    def _mapping(self) -> dict[str, str]:
        return {
            "CPSE": "Owner Org",
            "CPSE Material Code": "Part No",
            "Material Category": "Item Class",
            "Raw Description": "Long Text",
        }

    def test_renders_one_row_with_the_mapped_values(self):
        line = preview_mapped_row(_realistic_upload(), self._mapping(), index=0)
        assert line.startswith("Company: NTPC")
        assert "Code: NTPC-1000" in line
        assert "Category: Gate Valve" in line
        assert "Description: GATE V/V 100 NB CL-150 CS BODY BOLTED BONNET" in line

    def test_unmapped_field_is_labelled_not_omitted(self):
        mapping = {**self._mapping(), "Material Category": None}
        assert "Category: (unmapped)" in preview_mapped_row(
            _realistic_upload(), mapping, 0
        )

    def test_index_is_clamped_into_range(self):
        line = preview_mapped_row(_realistic_upload(n=5), self._mapping(), index=999)
        assert line  # no IndexError, returns the last row

    def test_empty_frame_gives_empty_string(self):
        assert preview_mapped_row(pd.DataFrame(), self._mapping()) == ""


class TestBuildDatasetFromMapped:
    def _mapped(self) -> pd.DataFrame:
        raw = _uploaded_frame()
        mapping = {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Material Code",
            "Material Category": "Category",
            "Raw Description": "Description",
        }
        return apply_column_mapping(raw, mapping)

    def test_has_labels_is_always_false(self):
        dataset = build_dataset_from_mapped(self._mapped(), "upload.csv")
        assert dataset.has_labels is False

    def test_eval_df_and_sector_reference_are_empty(self):
        dataset = build_dataset_from_mapped(self._mapped(), "upload.csv")
        assert dataset.eval_df.columns.empty
        assert dataset.sector_reference.empty

    def test_pipeline_df_is_usable(self):
        dataset = build_dataset_from_mapped(self._mapped(), "upload.csv")
        assert len(dataset.pipeline_df) == 2
        assert dataset.profile.n_rows == 2

    def test_leakage_guard_holds(self):
        dataset = build_dataset_from_mapped(self._mapped(), "upload.csv")
        dataset.assert_no_leakage()  # must not raise


class TestEstimateRuntimeSeconds:
    def test_baseline_matches_configured_seconds(self):
        estimate = estimate_runtime_seconds(config.RUNTIME_BASELINE_ROWS)
        assert estimate == pytest.approx(config.RUNTIME_BASELINE_SECONDS)

    def test_grows_faster_than_linearly(self):
        """Doubling the rows must more than double the estimate."""
        base = estimate_runtime_seconds(config.RUNTIME_BASELINE_ROWS)
        doubled = estimate_runtime_seconds(config.RUNTIME_BASELINE_ROWS * 2)
        assert doubled > base * 2

    def test_smaller_dataset_estimates_less_time(self):
        base = estimate_runtime_seconds(config.RUNTIME_BASELINE_ROWS)
        smaller = estimate_runtime_seconds(config.RUNTIME_BASELINE_ROWS // 2)
        assert smaller < base
