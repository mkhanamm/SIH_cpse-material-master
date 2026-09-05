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
    estimate_runtime_seconds,
    read_upload,
    suggest_column_mapping,
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
