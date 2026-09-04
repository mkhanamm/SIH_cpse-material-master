"""
Two-channel similarity computation with score fusion (Spec section 4.5).

WHAT THIS FILE DOES
    Scores every candidate pair on two independent channels and fuses them,
    following the research finding that the channels catch *different* duplicate
    types and neither dominates:

      SEMANTIC CHANNEL  - sentence embeddings + cosine similarity. Catches
        paraphrase and word reordering ("annunciator panel 0.415kv rated" vs
        "annunciator panel for 0.415 kv system").
        Pluggable backend, chosen in config.SEMANTIC_BACKEND:
          * "sbert"     - sentence-transformers, production default.
          * "tfidf_svd" - offline scikit-learn fallback (TF-IDF word+char
                          n-grams -> TruncatedSVD -> cosine). No model download
                          required, so the repo runs on an air-gapped machine.
          * "auto"      - sbert if importable and loadable, else tfidf_svd.
        Both backends are live and interchangeable. Whichever ran is recorded in
        `SemanticEncoder.backend_used` and surfaced in every evaluation report,
        so no number is ever ambiguous about which encoder produced it.

      STRING/ATTRIBUTE CHANNEL - Jaro-Winkler + token-set ratio on normalized
        text, plus structured attribute agreement (exact match on grade/spec,
        numeric-tolerance match on dimensions). Catches abbreviation and unit
        variants ("bfly v/v" vs "butterfly valve").

    Fusion is a configurable weighted sum (config.CHANNEL_WEIGHTS). Attribute
    *conflicts* (e.g. Schedule 20 vs Schedule 40) apply a penalty rather than
    merely lowering the average -- in this dataset the hard negatives are
    near-identical strings differing only in one spec value, so a conflicting
    attribute must be able to veto a high text score.

INPUTS
    Normalized text, extracted attributes, candidate pair list.

OUTPUTS
    SimilarityScores - per-pair dataclass: semantic, string, attribute,
                       fused, plus per-attribute agreement flags for the
                       explanation layer.

KEY FUNCTIONS
    SemanticEncoder(backend).encode(texts) -> ndarray
    semantic_similarity(pairs, embeddings) -> ndarray
    string_similarity(a, b)                -> float
    attribute_similarity(attrs_a, attrs_b) -> AttributeAgreement
    score_pairs(df, pairs)                 -> pandas.DataFrame
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import token_set_ratio
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from . import config
from .attribute_extraction import (
    CATEGORICAL_ATTRIBUTES,
    NUMERIC_ATTRIBUTES,
    _values_agree,
)

# Attribute agreement outcomes. "unknown" is a distinct third state and is
# deliberately NOT folded into "conflict": a missing spec on both sides is an
# absence of evidence, not evidence of difference. Treating it as a mismatch
# would penalise the 4,342 records with no Specification/Standard for the sole
# offence of being under-documented.
AGREE, CONFLICT, UNKNOWN = "agree", "conflict", "unknown"


# ---------------------------------------------------------------------------
# Semantic channel
# ---------------------------------------------------------------------------
class SemanticEncoder:
    """Sentence encoder with a production and an offline backend.

    The two backends produce different absolute score distributions, so a
    threshold tuned on one is not valid for the other. The trained classifier in
    `classifier.py` absorbs this automatically by learning its own weights on
    whichever backend produced the features; a hand-set threshold would not.

    Attributes:
        requested: The backend asked for.
        backend_used: The backend actually in use after resolution. Always
            report this alongside any metric.
    """

    def __init__(self, backend: str | None = None) -> None:
        """Resolve and initialise a backend.

        Args:
            backend: ``"sbert"``, ``"tfidf_svd"`` or ``"auto"``. Defaults to
                ``config.SEMANTIC_BACKEND``.

        Raises:
            RuntimeError: If ``"sbert"`` was demanded explicitly and cannot be
                loaded. ``"auto"`` never raises; it falls back silently but
                records the fallback in ``backend_used``.
        """
        self.requested = backend or config.SEMANTIC_BACKEND
        self._model = None
        self._pipeline = None
        self.backend_used = self._resolve(self.requested)

    def _resolve(self, backend: str) -> str:
        """Load the requested backend, falling back when permitted.

        Args:
            backend: Requested backend name.

        Returns:
            The backend name actually loaded.

        Raises:
            RuntimeError: If an explicit ``"sbert"`` request cannot be honoured.
            ValueError: On an unrecognised backend name.
        """
        if backend in ("sbert", "auto"):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(config.SBERT_MODEL_NAME)
                return "sbert"
            except Exception as exc:  # ImportError, network failure, OOM
                if backend == "sbert":
                    raise RuntimeError(
                        f"SBERT backend requested but unavailable: {exc}. "
                        "Install sentence-transformers and allow the model "
                        "download, or set config.SEMANTIC_BACKEND='tfidf_svd'."
                    ) from exc
                warnings.warn(
                    f"SBERT unavailable ({type(exc).__name__}); falling back to "
                    "the offline tfidf_svd encoder. Metrics will be labelled "
                    "with backend='tfidf_svd'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return "tfidf_svd"

        if backend == "tfidf_svd":
            return "tfidf_svd"

        raise ValueError(
            f"Unknown semantic backend {backend!r}; expected 'sbert', "
            "'tfidf_svd' or 'auto'."
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed normalized descriptions as unit-length vectors.

        Vectors are L2-normalized by both backends so cosine similarity is a
        plain dot product downstream.

        Args:
            texts: Normalized description strings.

        Returns:
            Array of shape ``(len(texts), dim)``.
        """
        if self.backend_used == "sbert":
            return np.asarray(
                self._model.encode(
                    texts, batch_size=64, show_progress_bar=False,
                    normalize_embeddings=True,
                )
            )
        return self._encode_tfidf_svd(texts)

    def _encode_tfidf_svd(self, texts: list[str]) -> np.ndarray:
        """Offline encoder: word + character TF-IDF reduced by SVD.

        Character n-grams are included alongside word n-grams because material
        descriptions are dense with codes ("a106", "e250", "m12") where subword
        overlap carries real signal that a pure word model discards.

        The SVD step is what makes this a *semantic-ish* channel rather than a
        second string channel: projecting into a dense latent space lets two
        descriptions with no shared surface tokens still score highly if they
        co-occur with the same vocabulary across the corpus.

        Args:
            texts: Normalized description strings.

        Returns:
            L2-normalized array of shape ``(len(texts), n_components)``.
        """
        if self._pipeline is None:
            n_components = min(config.TFIDF_SVD_COMPONENTS, max(2, len(texts) - 1))
            self._pipeline = make_pipeline(
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                    min_df=1,
                ),
                TruncatedSVD(
                    n_components=n_components, random_state=config.RANDOM_SEED
                ),
                Normalizer(copy=False),
            )
            word_vectors = self._pipeline.fit_transform(texts)
        else:
            word_vectors = self._pipeline.transform(texts)

        if self._char_pipeline_needed():
            char_vectors = self._encode_char(texts)
            combined = np.hstack([word_vectors, char_vectors])
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            return combined / np.clip(norms, 1e-9, None)
        return word_vectors

    def _char_pipeline_needed(self) -> bool:
        """Whether the character-n-gram half of the offline encoder is enabled.

        Returns:
            True -- kept as a named hook so the character half can be disabled
            for ablation without editing the encoder body.
        """
        return True

    def _encode_char(self, texts: list[str]) -> np.ndarray:
        """Character n-gram half of the offline encoder.

        Args:
            texts: Normalized description strings.

        Returns:
            Reduced character-level vectors.
        """
        if getattr(self, "_char_pipeline", None) is None:
            n_components = min(
                config.TFIDF_SVD_COMPONENTS // 2, max(2, len(texts) - 1)
            )
            self._char_pipeline = make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                    min_df=2,
                ),
                TruncatedSVD(
                    n_components=n_components, random_state=config.RANDOM_SEED
                ),
                Normalizer(copy=False),
            )
            return self._char_pipeline.fit_transform(texts)
        return self._char_pipeline.transform(texts)


def semantic_similarity(
    pairs: list[tuple[int, int]],
    embeddings: np.ndarray,
    positions: dict[int, int] | None = None,
) -> np.ndarray:
    """Cosine similarity for each candidate pair, clipped to [0, 1].

    Args:
        pairs: Candidate index pairs.
        embeddings: Unit-length embeddings from :meth:`SemanticEncoder.encode`.
        positions: Optional map from dataframe index to embedding row. Required
            when the frame index is not 0..n-1.

    Returns:
        Array of similarities aligned with ``pairs``.
    """
    if not pairs:
        return np.zeros(0)
    if positions is None:
        left = np.array([a for a, _ in pairs])
        right = np.array([b for _, b in pairs])
    else:
        left = np.array([positions[a] for a, _ in pairs])
        right = np.array([positions[b] for _, b in pairs])

    sims = np.einsum("ij,ij->i", embeddings[left], embeddings[right])
    # Cosine is in [-1, 1]; negatives carry no meaning for "same material" and
    # would otherwise drag the fused score below zero.
    return np.clip(sims, 0.0, 1.0)


# ---------------------------------------------------------------------------
# String channel
# ---------------------------------------------------------------------------
def string_similarity(a: str, b: str) -> float:
    """Blend Jaro-Winkler and token-set ratio on normalized text.

    The two are complementary: Jaro-Winkler rewards shared prefixes and is
    sensitive to character-level edits (abbreviation and unit variants), while
    token-set ratio is order-independent and handles the reordering that is
    endemic to material descriptions ("bearing deep groove ball bore 25 mm" vs
    "deep groove ball bearing 25 mm bore").

    Args:
        a: First normalized description.
        b: Second normalized description.

    Returns:
        Similarity in [0, 1].
    """
    if not a or not b:
        return 0.0
    jw = JaroWinkler.normalized_similarity(a, b)
    tsr = token_set_ratio(a, b) / 100.0
    return 0.5 * jw + 0.5 * tsr


# ---------------------------------------------------------------------------
# Attribute channel
# ---------------------------------------------------------------------------
@dataclass
class AttributeAgreement:
    """Per-attribute comparison of two records.

    Attributes:
        flags: Attribute name -> ``agree`` | ``conflict`` | ``unknown``.
        n_agree: Attributes both records state and agree on.
        n_conflict: Attributes both records state and disagree on.
        n_comparable: ``n_agree + n_conflict`` -- the count of attributes where
            evidence actually existed.
        score: Agreement ratio in [0, 1], or 0.5 (neutral) when nothing was
            comparable. Neutral, not zero: absence of evidence must not read as
            evidence of difference.
        conflicting: Names of the conflicting attributes, for the explanation.
    """

    flags: dict[str, str] = field(default_factory=dict)
    n_agree: int = 0
    n_conflict: int = 0
    n_comparable: int = 0
    score: float = 0.5
    conflicting: list[str] = field(default_factory=list)

    def has_hard_conflict(self) -> bool:
        """Whether a conflict occurred on an attribute that must match exactly.

        Grade, spec, schedule and pressure class are procurement-critical: a
        Schedule 20 pipe cannot be issued against a Schedule 40 requisition,
        however similar the descriptions read.

        Returns:
            True if any conflicting attribute is in
            ``config.EXACT_MATCH_ATTRIBUTES``.
        """
        return any(a in config.EXACT_MATCH_ATTRIBUTES for a in self.conflicting)


def _is_missing(value: object) -> bool:
    """Test whether an extracted attribute value is absent.

    Args:
        value: An attribute value.

    Returns:
        True for None, NaN, or empty/whitespace strings.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def attribute_similarity(
    attrs_a: dict[str, object], attrs_b: dict[str, object]
) -> AttributeAgreement:
    """Compare two records attribute by attribute.

    Numeric attributes match within ``config.DIMENSION_TOLERANCE_PCT``;
    categorical attributes match on containment-aware string equality. Any
    attribute missing on either side is recorded as ``unknown`` and excluded
    from the score rather than counted against the pair.

    Args:
        attrs_a: Extracted attributes of the first record.
        attrs_b: Extracted attributes of the second record.

    Returns:
        A populated :class:`AttributeAgreement`.
    """
    flags: dict[str, str] = {}
    conflicting: list[str] = []
    n_agree = n_conflict = 0

    for attribute in NUMERIC_ATTRIBUTES + CATEGORICAL_ATTRIBUTES:
        va, vb = attrs_a.get(attribute), attrs_b.get(attribute)
        if _is_missing(va) or _is_missing(vb):
            flags[attribute] = UNKNOWN
            continue
        if _values_agree(va, vb):
            flags[attribute] = AGREE
            n_agree += 1
        else:
            flags[attribute] = CONFLICT
            conflicting.append(attribute)
            n_conflict += 1

    comparable = n_agree + n_conflict
    score = (n_agree / comparable) if comparable else 0.5

    return AttributeAgreement(
        flags=flags,
        n_agree=n_agree,
        n_conflict=n_conflict,
        n_comparable=comparable,
        score=score,
        conflicting=conflicting,
    )


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def fuse_scores(
    semantic: float, string: float, agreement: AttributeAgreement
) -> float:
    """Combine the channels into one score, applying the conflict veto.

    The weighted sum alone is not enough. Two records reading
    "seamless carbon steel pipe 40 mm schedule 20" and
    "... schedule 40" score ~0.97 on both text channels; without a veto the
    attribute channel can only pull the fused score down by its weight, leaving
    the pair comfortably above the high-confidence threshold. The penalty makes
    a procurement-critical conflict decisive.

    Args:
        semantic: Semantic channel score in [0, 1].
        string: String channel score in [0, 1].
        agreement: Output of :func:`attribute_similarity`.

    Returns:
        Fused score in [0, 1].
    """
    weights = config.CHANNEL_WEIGHTS
    fused = (
        weights["semantic"] * semantic
        + weights["string"] * string
        + weights["attribute"] * agreement.score
    )
    if agreement.has_hard_conflict():
        fused -= config.ATTRIBUTE_CONFLICT_PENALTY
    elif agreement.n_conflict:
        # A soft conflict (e.g. differing head in metres) is real evidence but
        # not disqualifying; charge a fraction of the penalty.
        fused -= config.ATTRIBUTE_CONFLICT_PENALTY * 0.5
    return float(np.clip(fused, 0.0, 1.0))


def score_pairs(
    df: pd.DataFrame,
    pairs: list[tuple[int, int]],
    normalized_text: pd.Series,
    encoder: SemanticEncoder | None = None,
    attribute_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Score every candidate pair on all channels.

    Args:
        df: Frame joining pipeline columns with extracted attributes.
        pairs: Candidate index pairs from ``blocking.candidate_pairs``.
        normalized_text: Normalized descriptions, indexed like ``df``.
        encoder: Reusable encoder; one is constructed if omitted.
        attribute_columns: Attribute names to compare; defaults to all
            extracted attributes.

    Returns:
        One row per pair with columns: ``idx_a``, ``idx_b``, ``semantic``,
        ``string``, ``attribute``, ``fused``, ``n_comparable_attrs``,
        ``n_conflicts``, ``hard_conflict``, ``min_known_attrs``,
        ``same_category``, ``cross_cpse``, and ``flags`` (dict) for the
        explanation layer.
    """
    encoder = encoder or SemanticEncoder()
    attribute_columns = attribute_columns or (
        NUMERIC_ATTRIBUTES + CATEGORICAL_ATTRIBUTES
    )

    texts = normalized_text.reindex(df.index).fillna("").tolist()
    embeddings = encoder.encode(texts)
    positions = {idx: pos for pos, idx in enumerate(df.index)}

    semantic_scores = semantic_similarity(pairs, embeddings, positions)

    attrs_lookup = {
        idx: {c: row.get(c) for c in attribute_columns}
        for idx, row in df[attribute_columns].iterrows()
    }
    n_known = {
        idx: sum(1 for v in values.values() if not _is_missing(v))
        for idx, values in attrs_lookup.items()
    }
    text_lookup = dict(zip(df.index, texts))
    categories = df["Material Category"].to_dict()
    cpses = df["CPSE"].to_dict()

    records: list[dict[str, object]] = []
    for position, (a, b) in enumerate(pairs):
        agreement = attribute_similarity(attrs_lookup[a], attrs_lookup[b])
        string_score = string_similarity(text_lookup[a], text_lookup[b])
        semantic_score = float(semantic_scores[position])
        records.append(
            {
                "idx_a": a,
                "idx_b": b,
                "semantic": semantic_score,
                "string": string_score,
                "attribute": agreement.score,
                "fused": fuse_scores(semantic_score, string_score, agreement),
                "n_comparable_attrs": agreement.n_comparable,
                "n_agree_attrs": agreement.n_agree,
                "n_conflicts": agreement.n_conflict,
                "hard_conflict": agreement.has_hard_conflict(),
                "min_known_attrs": min(n_known[a], n_known[b]),
                "same_category": categories[a] == categories[b],
                "cross_cpse": cpses[a] != cpses[b],
                "flags": agreement.flags,
            }
        )

    scored = pd.DataFrame.from_records(records)
    scored.attrs["semantic_backend"] = encoder.backend_used
    return scored


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from . import attribute_extraction, blocking, ingestion
    from .normalization import normalize_series

    dataset = ingestion.load_dataset()
    extracted = attribute_extraction.extract_frame(dataset.pipeline_df)
    joined = dataset.pipeline_df.join(extracted)
    text = normalize_series(joined[config.INPUT_TEXT_COLUMN])

    result = blocking.candidate_pairs(joined)
    scored = score_pairs(joined, result.pairs[:2000], text)
    print(f"backend = {scored.attrs['semantic_backend']}")
    print(scored[["semantic", "string", "attribute", "fused"]].describe())
