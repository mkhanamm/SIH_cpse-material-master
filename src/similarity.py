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
