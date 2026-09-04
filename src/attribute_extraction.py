"""
Rule-based structured attribute extraction (Spec section 4.3).

WHAT THIS FILE DOES
    Pulls structured technical attributes out of normalized description text
    using deterministic, inspectable regex rules -- deliberately NOT a model, so
    a reviewer can always see exactly why a value was extracted.

    Extracted attributes: nominal_size_mm, thickness_mm, grade, spec_standard,
    schedule, pressure_class, capacity, rating_kva, voltage_kv, ratio, ply,
    material_of_construction.

    Where the dataset's own semi-structured columns (`Material/Grade`,
    `Dimensions`, `Specification/Standard`, `Capacity/Rating`,
    `Operating Parameter`) are populated, they are merged in and used to
    cross-validate the regex output (see `validate_against_columns`).

    Missing values degrade gracefully to None and are reported as
    "attribute unknown" -- never fabricated. Downstream, an unknown attribute
    contributes neutral evidence, not a match.

INPUTS
    Normalized description text + the row's semi-structured attribute columns.

OUTPUTS
    AttributeSet - dataclass of typed optional attributes plus .provenance
                   ({attribute: 'regex' | 'column' | 'both' | 'unknown'}).

KEY FUNCTIONS
    extract(text, row)                    -> AttributeSet
    extract_frame(df, text_col)           -> pandas.DataFrame
    validate_against_columns(df)          -> dict  (agreement rates, for the report)
"""
