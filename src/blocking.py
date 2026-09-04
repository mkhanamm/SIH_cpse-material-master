"""
Candidate blocking -- run before any expensive similarity (Spec section 4.4).

WHAT THIS FILE DOES
    Reduces the O(n^2) comparison space to a tractable candidate set by bucketing
    records on cheap keys, then emitting only within-bucket pairs.

    Two key families are used as a UNION, because neither alone is sufficient:
      - CATEGORY key:  `Material Category` (+ coarse size bucket when known).
        Cheap and precise, but caps recall -- 32 ground-truth groups in this
        dataset span more than one category (e.g. dragline vs shovel bucket).
      - ATTRIBUTE-SIGNATURE key: a coarse fingerprint of extracted attributes
        (rounded size, spec family, capacity band) that ignores category and so
        recovers most of the cross-category pairs the first key loses.

    Reports honest metrics: comparisons before/after, reduction factor, and the
    *recall ceiling* the blocking scheme imposes on the ground-truth pairs --
    a speedup number without its recall cost is not a real measurement.

INPUTS
    DataFrame with `Material Category` and extracted attribute columns.

OUTPUTS
    BlockingResult - .pairs (candidate index pairs), .blocks, .stats
    BlockingStats  - naive_comparisons, blocked_comparisons, reduction_factor,
                     recall_ceiling, pairs_lost

KEY FUNCTIONS
    build_blocks(df)                       -> dict[str, list[int]]
    candidate_pairs(df)                    -> BlockingResult
    evaluate_blocking(result, truth_pairs) -> BlockingStats
"""
