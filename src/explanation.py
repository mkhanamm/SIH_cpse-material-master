"""
Deterministic match explanation generator (Spec section 4.7).

WHAT THIS FILE DOES
    Renders a human-readable justification for every proposed match, from
    numbers already computed upstream -- no live LLM call.

    The output *structure* is modelled on LLM-based entity-matching explanation
    work (per-attribute alignment + per-channel scores + overall verdict), but
    the implementation is a fixed template. This is a deliberate engineering
    choice for demo reliability: no API key, no network dependency, no cost, no
    non-determinism, and identical output every time a judge clicks the button.
    Swapping in a live LLM for richer prose is a one-function change
    (`render_explanation` is the only seam) -- documented as an upgrade path,
    not hidden as a limitation.

    Example output:
        Match: HIGH confidence (fused score 0.91)
        + Material grade identical (ASTM A106)
        + Dimension match within tolerance (40 mm NB vs 40 mm NB)
        + Description similarity: semantic 0.88 / string 0.94
        - Specification differs (A106 vs A333)
        ? Operating parameter unknown for both records

INPUTS
    A scored pair or cluster + its SimilarityScores and AttributeSets.

OUTPUTS
    Explanation - .verdict, .lines, .as_text(), .as_dict()  (dict form is what
                  gets written to the audit trail, so explanations are
                  reproducible evidence, not UI-only decoration).

KEY FUNCTIONS
    explain_pair(pair, scores, attrs_a, attrs_b) -> Explanation
    explain_cluster(cluster)                     -> Explanation
    render_explanation(explanation)              -> str   (the LLM swap seam)
"""
