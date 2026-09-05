"""Unit tests for src.ingestion -- has_labels detection and ground-truth guards."""

from __future__ import annotations

import pandas as pd
import pytest

from src import ingestion


def _materials_frame(with_labels: bool) -> pd.DataFrame:
    """Minimal materials frame, with or without the ground-truth columns."""
    data = {
        "CPSE": ["ONGC", "BPCL", "ONGC", "IOCL"],
        "CPSE Material Code": ["M1", "M2", "M3", "M4"],
        "Material Category": ["Valve", "Valve", "Bearing", "Bearing"],
        "Sector": ["Oil & Gas"] * 4,
        "UOM": ["NOS", "NOS", "NOS", "NOS"],
        "Raw Description": [
            "BFLY V/V 300 CI",
            "BUTTERFLY VALVE 300 CAST IRON",
            "BRG DGB 25MM",
            "DEEP GROOVE BALL BEARING 25 MM",
        ],
    }
    if with_labels:
        data["Standardized Description"] = [
            "BUTTERFLY VALVE 300 CAST IRON",
            "BUTTERFLY VALVE 300 CAST IRON",
            "DEEP GROOVE BALL BEARING 25 MM",
            "DEEP GROOVE BALL BEARING 25 MM",
        ]
        data["GroundTruth_Group"] = ["G1", "G1", "G2", "G2"]
    return pd.DataFrame(data)


def _write_workbook(path, with_labels: bool) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _materials_frame(with_labels).to_excel(
            writer, sheet_name="All_Materials", index=False
        )


@pytest.fixture
def labelled_path(tmp_path):
    path = tmp_path / "labelled.xlsx"
    _write_workbook(path, with_labels=True)
    return path


@pytest.fixture
def unlabelled_path(tmp_path):
    path = tmp_path / "unlabelled.xlsx"
    _write_workbook(path, with_labels=False)
    return path


class TestHasLabels:
    def test_labelled_dataset_sets_flag_true(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        assert dataset.has_labels is True
        assert "GroundTruth_Group" in dataset.eval_df.columns

    def test_unlabelled_dataset_does_not_raise(self, unlabelled_path):
        """A real CPSE upload has no GroundTruth_Group; loading it must not crash."""
        dataset = ingestion.load_dataset(unlabelled_path)
        assert dataset.has_labels is False

    def test_unlabelled_eval_df_has_no_ground_truth_column(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        assert "GroundTruth_Group" not in dataset.eval_df.columns

    def test_unlabelled_pipeline_df_is_still_usable(self, unlabelled_path):
        """The matcher-visible columns must be unaffected by the missing label."""
        dataset = ingestion.load_dataset(unlabelled_path)
        assert len(dataset.pipeline_df) == 4
        assert "Raw Description" in dataset.pipeline_df.columns

    def test_leakage_guard_still_holds_when_unlabelled(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        dataset.assert_no_leakage()  # must not raise


class TestGroundTruthHelpersRequireLabels:
    def test_ground_truth_pairs_raises_clearly(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        with pytest.raises(ValueError, match="no ground-truth labels"):
            ingestion.ground_truth_pairs(dataset.eval_df)

    def test_ground_truth_summary_raises_clearly(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        with pytest.raises(ValueError, match="no ground-truth labels"):
            ingestion.ground_truth_summary(dataset.eval_df, dataset.pipeline_df)

    def test_find_cross_cpse_examples_raises_clearly(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        with pytest.raises(ValueError, match="no ground-truth labels"):
            ingestion.find_cross_cpse_examples(dataset)

    def test_ground_truth_pairs_works_when_labelled(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        pairs = ingestion.ground_truth_pairs(dataset.eval_df)
        assert pairs == {(0, 1), (2, 3)}


class TestFindCrossCpseExamples:
    """Examples feeding the 'The Problem' view's tables and filters."""

    def test_includes_sector_and_category_columns(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        examples = ingestion.find_cross_cpse_examples(dataset)
        assert examples
        for example in examples:
            assert "Sector" in example.columns
            assert "Material Category" in example.columns

    def test_limit_caps_the_number_of_examples(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        examples = ingestion.find_cross_cpse_examples(dataset, limit=1)
        assert len(examples) == 1

    def test_limit_none_returns_every_qualifying_group(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        examples = ingestion.find_cross_cpse_examples(dataset, limit=None)
        assert len(examples) == 2  # both G1 (rows 0,1) and G2 (rows 2,3) qualify


class TestSampleDataset:
    """A large upload must still be demoable -- see the 'Load Data' view."""

    def test_sample_reduces_row_count(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        sampled = ingestion.sample_dataset(dataset, 2, seed=0)
        assert len(sampled.pipeline_df) == 2

    def test_sample_reindexes_from_zero(self, unlabelled_path):
        """Downstream code uses positional index as the record id."""
        dataset = ingestion.load_dataset(unlabelled_path)
        sampled = ingestion.sample_dataset(dataset, 2, seed=0)
        assert list(sampled.pipeline_df.index) == [0, 1]

    def test_eval_df_stays_aligned_with_pipeline_df(self, labelled_path):
        """Sampling must not desynchronize a record from its label."""
        dataset = ingestion.load_dataset(labelled_path)
        sampled = ingestion.sample_dataset(dataset, 2, seed=0)
        assert len(sampled.eval_df) == len(sampled.pipeline_df)

    def test_n_larger_than_dataset_returns_dataset_unchanged(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        assert ingestion.sample_dataset(dataset, 1000) is dataset

    def test_n_below_one_is_clamped(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        sampled = ingestion.sample_dataset(dataset, 0, seed=0)
        assert len(sampled.pipeline_df) == 1

    def test_profile_recomputed_on_the_sample(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        sampled = ingestion.sample_dataset(dataset, 2, seed=0)
        assert sampled.profile.n_rows == 2

    def test_has_labels_and_source_path_carried_over(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        sampled = ingestion.sample_dataset(dataset, 2, seed=0)
        assert sampled.has_labels == dataset.has_labels
        assert sampled.source_path == dataset.source_path

    def test_sample_is_reproducible_with_the_same_seed(self, unlabelled_path):
        dataset = ingestion.load_dataset(unlabelled_path)
        first = ingestion.sample_dataset(dataset, 2, seed=7)
        second = ingestion.sample_dataset(dataset, 2, seed=7)
        assert list(first.pipeline_df["CPSE Material Code"]) == list(
            second.pipeline_df["CPSE Material Code"]
        )

    def test_leakage_guard_still_holds_after_sampling(self, labelled_path):
        dataset = ingestion.load_dataset(labelled_path)
        sampled = ingestion.sample_dataset(dataset, 2, seed=0)
        sampled.assert_no_leakage()  # must not raise
