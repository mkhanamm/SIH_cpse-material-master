"""Unit tests for src.similarity -- channel bounds, symmetry, attribute-conflict veto behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.similarity import (
    AGREE,
    CONFLICT,
    UNKNOWN,
    AttributeAgreement,
    SemanticEncoder,
    attribute_similarity,
    fuse_scores,
    semantic_similarity,
    string_similarity,
)


class TestStringChannel:
    def test_identical_text_scores_one(self):
        assert string_similarity("butterfly valve 300", "butterfly valve 300") == 1.0

    def test_is_symmetric(self):
        a, b = "bearing deep groove ball 25 mm", "deep groove ball bearing 25 mm"
        assert string_similarity(a, b) == pytest.approx(string_similarity(b, a))

    def test_bounded_to_unit_interval(self):
        for a, b in [("pipe", "transformer"), ("a", "a"), ("", "pipe")]:
            assert 0.0 <= string_similarity(a, b) <= 1.0

    def test_handles_empty_input(self):
        assert string_similarity("", "pipe") == 0.0

    def test_reordering_scores_highly(self):
        """Token-set ratio is the half of the blend that must carry this."""
        score = string_similarity(
            "bearing deep groove ball bore 25 mm",
            "deep groove ball bearing 25 mm bore",
        )
        assert score > 0.85

    def test_different_materials_score_lower_than_variants(self):
        variant = string_similarity("butterfly valve lug 300", "butterfly valve 300 lug")
        unrelated = string_similarity("butterfly valve lug 300", "distribution transformer 63")
        assert variant > unrelated


class TestSemanticChannel:
    def test_offline_backend_always_available(self):
        """The repo must run with no model download and no network."""
        assert SemanticEncoder("tfidf_svd").backend_used == "tfidf_svd"

    def test_auto_backend_resolves_to_something_usable(self):
        assert SemanticEncoder("auto").backend_used in {"sbert", "tfidf_svd"}

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown semantic backend"):
            SemanticEncoder("word2vec")

    def test_embeddings_are_unit_length(self):
        """Cosine is computed as a plain dot product, which assumes this."""
        encoder = SemanticEncoder("tfidf_svd")
        vectors = encoder.encode(
            [
                "butterfly valve lug 300 nominal bore class 150",
                "butterfly valve lug type 300 mm class 150",
                "distribution transformer 63 kva",
                "carbon steel seamless pipe 40 mm schedule 20",
            ]
        )
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_similarity_is_clipped_to_unit_interval(self):
        encoder = SemanticEncoder("tfidf_svd")
        vectors = encoder.encode(["pipe 40 mm", "transformer 63 kva", "pipe 40 mm"])
        scores = semantic_similarity([(0, 1), (0, 2)], vectors)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_identical_text_is_maximally_similar(self):
        encoder = SemanticEncoder("tfidf_svd")
        vectors = encoder.encode(["pipe 40 mm schedule 20", "transformer 63 kva", "pipe 40 mm schedule 20"])
        scores = semantic_similarity([(0, 2), (0, 1)], vectors)
        assert scores[0] > scores[1]

    def test_empty_pair_list(self):
        assert semantic_similarity([], np.zeros((3, 4))).shape == (0,)


class TestAttributeChannel:
    def test_agreeing_attributes(self):
        result = attribute_similarity(
            {"nominal_size_mm": 40.0, "schedule": "20"},
            {"nominal_size_mm": 40.0, "schedule": "20"},
        )
        assert result.n_agree == 2
        assert result.n_conflict == 0
        assert result.score == 1.0

    def test_conflicting_schedule_is_a_hard_conflict(self):
        result = attribute_similarity(
            {"nominal_size_mm": 40.0, "schedule": "20"},
            {"nominal_size_mm": 40.0, "schedule": "40"},
        )
        assert result.flags["schedule"] == CONFLICT
        assert result.has_hard_conflict()

    def test_missing_attribute_is_unknown_not_conflict(self):
        """Absence of evidence must not read as evidence of difference."""
        result = attribute_similarity(
            {"nominal_size_mm": 40.0, "schedule": "20"},
            {"nominal_size_mm": 40.0},
        )
        assert result.flags["schedule"] == UNKNOWN
        assert result.n_conflict == 0
        assert result.score == 1.0

    def test_no_comparable_attributes_scores_neutral(self):
        """Neutral 0.5, not 0.0 -- two bare descriptions are undecided, not different."""
        result = attribute_similarity({}, {})
        assert result.score == 0.5
        assert result.n_comparable == 0

    def test_numeric_tolerance_applied(self):
        result = attribute_similarity(
            {"nominal_size_mm": 40.0}, {"nominal_size_mm": 40.5}
        )
        assert result.flags["nominal_size_mm"] == AGREE

    def test_soft_conflict_is_not_hard(self):
        result = attribute_similarity({"head_m": 20.0}, {"head_m": 100.0})
        assert result.n_conflict == 1
        assert not result.has_hard_conflict()

    def test_nan_treated_as_missing(self):
        result = attribute_similarity(
            {"nominal_size_mm": float("nan")}, {"nominal_size_mm": 40.0}
        )
        assert result.flags["nominal_size_mm"] == UNKNOWN


class TestFusion:
    def test_perfect_match_scores_high(self):
        agreement = attribute_similarity(
            {"nominal_size_mm": 40.0, "schedule": "20"},
            {"nominal_size_mm": 40.0, "schedule": "20"},
        )
        assert fuse_scores(1.0, 1.0, agreement) == pytest.approx(1.0)

    def test_hard_conflict_vetoes_near_identical_text(self):
        """The Schedule 20 vs 40 case: 0.97 text similarity must not survive."""
        agreement = attribute_similarity(
            {"nominal_size_mm": 40.0, "schedule": "20"},
            {"nominal_size_mm": 40.0, "schedule": "40"},
        )
        without_conflict = fuse_scores(
            0.97, 0.97, AttributeAgreement(score=agreement.score)
        )
        with_conflict = fuse_scores(0.97, 0.97, agreement)
        assert with_conflict < without_conflict
        assert without_conflict - with_conflict == pytest.approx(
            config.ATTRIBUTE_CONFLICT_PENALTY
        )

    def test_soft_conflict_charges_half_penalty(self):
        agreement = attribute_similarity({"head_m": 20.0}, {"head_m": 100.0})
        baseline = fuse_scores(0.9, 0.9, AttributeAgreement(score=agreement.score))
        penalised = fuse_scores(0.9, 0.9, agreement)
        assert baseline - penalised == pytest.approx(
            config.ATTRIBUTE_CONFLICT_PENALTY * 0.5
        )

    def test_output_bounded(self):
        agreement = attribute_similarity(
            {"schedule": "20"}, {"schedule": "40"}
        )
        assert 0.0 <= fuse_scores(0.0, 0.0, agreement) <= 1.0
        assert 0.0 <= fuse_scores(1.0, 1.0, agreement) <= 1.0

    def test_weights_sum_to_one(self):
        """Otherwise a 'perfect' pair cannot reach 1.0 and thresholds mislead."""
        assert sum(config.CHANNEL_WEIGHTS.values()) == pytest.approx(1.0)
