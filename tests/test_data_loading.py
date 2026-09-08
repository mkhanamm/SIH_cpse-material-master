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
    match_column_headers,
    preview_mapped_row,
    read_upload,
    suggest_column_mapping,
)


def _real_world_upload() -> pd.DataFrame:
    """A 10-row export with the header set that broke the shape-only guesser.

    Headers: Enterprise, Part Number, Item Category, Item Description,
    Material Grade, Size, Standard, Rating, Unit -- where the company,
    grade and unit-of-measure columns are all short repeated labels that a
    distinct-count heuristic cannot tell apart.
    """
    companies = ["NTPC", "SAIL", "BHEL", "NTPC", "GAIL", "SAIL", "IOCL", "BHEL",
                 "ONGC", "NMDC"]
    return pd.DataFrame(
        {
            "Enterprise": companies,
            "Part Number": [f"{c}-{45001 + i}" for i, c in enumerate(companies)],
            "Item Category": [
                "Gate Valve", "Ball Bearing", "Seamless Pipe", "Gate Valve",
                "Gasket", "Ball Bearing", "Centrifugal Pump", "Transformer",
                "Conveyor Idler", "Circuit Breaker",
            ],
            "Item Description": [
                "GATE VALVE 100 NB CLASS 150 CARBON STEEL BODY BOLTED BONNET",
                "DEEP GROOVE BALL BEARING 6205 25 MM BORE C3 CLEARANCE",
                "SEAMLESS CARBON STEEL PIPE 40 NB SCH 40 ASTM A106 GR B",
                "GATE VALVE 100 NB CLASS 150 CAST STEEL BODY RISING STEM",
                "SPIRAL WOUND GASKET 150 NB CLASS 300 SS316 GRAPHITE FILLER",
                "DEEP GROOVE BALL BEARING 6205 25 MM BORE ZZ SHIELDED",
                "CENTRIFUGAL PUMP 50 M3/HR 30 M HEAD 15 KW 2900 RPM",
                "DISTRIBUTION TRANSFORMER 63 KVA 11/0.433 KV ONAN COPPER",
                "TROUGHING IDLER ROLLER 152 MM DIA 3 ROLL CARRYING SET",
                "VACUUM CIRCUIT BREAKER 12 KV 1250 A 25 KA WITHDRAWABLE",
            ],
            "Material Grade": ["A216 WCB", "SS316", "A106 GR B", "A216 WCB",
                               "SS316", "SS316", "CI", "CRGO", "EN8", "EPDM"],
            "Size": ["100 NB", "25 MM", "40 NB", "100 NB", "150 NB", "25 MM",
                     "50 NB", "1000 KVA", "900 MM", "12 KV"],
            "Standard": ["API 600", "SKF 6205", "ASTM A106", "API 600",
                         "ASME B16.20", "SKF 6205", "IS 1520", "IS 2026",
                         "IS 8598", "IS 13118"],
            "Rating": ["PN 16", "", "SCH 40", "PN 16", "300#", "", "32 M",
                       "63 KVA", "", ""],
            "Unit": ["NOS", "NOS", "MTR", "NOS", "NOS", "NOS", "NOS", "NOS",
                     "SET", "NOS"],
        }
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

    def test_falls_back_to_data_shape_when_headers_are_opaque(self):
        df = _realistic_upload().rename(
            columns={
                "Owner Org": "F1", "Part No": "F2", "Item Class": "F3",
                "Long Text": "F4", "UoM": "F5",
            }
        )
        inferred = infer_column_mapping(df)
        assert inferred["Raw Description"] == "F4"      # longest average text
        assert inferred["CPSE Material Code"] == "F2"   # unique + dash-segmented
        assert inferred["CPSE"] == "F1"                 # fewer distinct labels
        assert inferred["Material Category"] == "F3"

    def test_low_confidence_required_field_is_left_unmapped(self):
        """One ambiguous label column -> CPSE stays None, not a bad guess."""
        df = pd.DataFrame(
            {
                "desc_txt": [
                    f"a long free-text material description number {i}" for i in range(8)
                ],
                "code": [f"MC-{i:04d}" for i in range(8)],
                "grp": ["Valve", "Pump", "Valve", "Pipe", "Pump", "Valve",
                        "Pipe", "Gasket"],
                "uom": ["NOS"] * 8,
            }
        )
        inferred = infer_column_mapping(df)
        assert inferred["CPSE"] is None

    def test_optional_fields_are_not_filled_from_leftover_columns(self):
        df = _real_world_upload()
        inferred = infer_column_mapping(df)
        assert inferred["Sector"] is None
        assert inferred["Operating Parameter"] is None


class TestInferColumnMappingRealWorldHeaders:
    """The exact header set from the bug report:
    Enterprise, Part Number, Item Category, Item Description, Material Grade,
    Size, Standard, Rating, Unit.
    """

    def test_produces_the_correct_required_mapping(self):
        inferred = infer_column_mapping(_real_world_upload())
        assert inferred["CPSE"] == "Enterprise"
        assert inferred["CPSE Material Code"] == "Part Number"
        assert inferred["Material Category"] == "Item Category"
        assert inferred["Raw Description"] == "Item Description"

    def test_unit_of_measure_column_is_never_company_or_category(self):
        inferred = infer_column_mapping(_real_world_upload())
        assert inferred["CPSE"] != "Unit"
        assert inferred["Material Category"] != "Unit"

    def test_material_grade_column_does_not_become_material_category(self):
        inferred = infer_column_mapping(_real_world_upload())
        assert inferred["Material Category"] != "Material Grade"

    def test_sector_is_not_guessed_from_the_company_column(self):
        assert infer_column_mapping(_real_world_upload())["Sector"] is None

    def test_optional_attribute_columns_map_by_fuzzy_header(self):
        inferred = infer_column_mapping(_real_world_upload())
        assert inferred["Material/Grade"] == "Material Grade"
        assert inferred["Dimensions"] == "Size"
        assert inferred["Specification/Standard"] == "Standard"
        assert inferred["Capacity/Rating"] == "Rating"

    def test_full_mapping_is_exactly_as_expected(self):
        assert infer_column_mapping(_real_world_upload()) == {
            "CPSE": "Enterprise",
            "CPSE Material Code": "Part Number",
            "Material Category": "Item Category",
            "Raw Description": "Item Description",
            "Sector": None,
            "Material/Grade": "Material Grade",
            "Dimensions": "Size",
            "Specification/Standard": "Standard",
            "Capacity/Rating": "Rating",
            "Operating Parameter": None,
        }


class TestMatchColumnHeaders:
    def test_fuzzy_synonyms_map_to_the_right_field(self):
        df = pd.DataFrame(
            {
                "Company Name": ["NTPC", "SAIL"],
                "Organisation": ["A", "B"],
                "Item Group": ["Valve", "Pump"],
                "Long Description": ["x y z", "p q r"],
            }
        )
        matched = match_column_headers(df)
        assert matched["CPSE"] in {"Company Name", "Organisation"}
        assert matched["Material Category"] == "Item Group"
        assert matched["Raw Description"] == "Long Description"

    def test_below_threshold_header_is_left_none(self):
        df = pd.DataFrame({"Widget": ["a", "b"], "Blob": ["c", "d"]})
        assert set(match_column_headers(df).values()) == {None}

    def test_uom_column_is_not_matched_to_cpse_even_with_a_matching_header(self):
        # Header "Unit" fuzzily matches CPSE's "unit name" synonym, but the
        # values are unit-of-measure tokens.
        df = pd.DataFrame(
            {"Unit": ["NOS", "MTR", "NOS", "SET"], "Plant": ["NTPC", "SAIL", "BHEL", "GAIL"]}
        )
        matched = match_column_headers(df)
        assert matched["CPSE"] == "Plant"

    def test_covers_every_mappable_target(self):
        matched = match_column_headers(_real_world_upload())
        assert set(matched) == set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)


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

    def test_tiny_dataset_never_estimates_zero(self):
        """A ten-row file still pays encoder/classifier setup -- not '~0s'."""
        estimate = estimate_runtime_seconds(10)
        assert estimate >= config.RUNTIME_FIXED_OVERHEAD_SECONDS
        assert round(estimate) > 0

    def test_fixed_overhead_dominates_for_the_smallest_files(self):
        assert estimate_runtime_seconds(1) == pytest.approx(
            config.RUNTIME_FIXED_OVERHEAD_SECONDS, abs=0.5
        )
