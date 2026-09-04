"""
NLP normalization of free-text material descriptions (Spec section 4.2).

WHAT THIS FILE DOES
    Turns a messy `Raw Description` such as `BFLY V/V LUG 300 NB CL-150` into a
    canonical, comparable form: `butterfly valve lug 300 nominal bore class 150`.

    Three ordered passes, each independently inspectable:
      1. Case folding, punctuation and whitespace repair.
      2. Abbreviation expansion from an editable lookup table (ABBREVIATIONS),
         seeded by inspecting real `Raw Description` values in this dataset.
      3. Unit canonicalization (inch->mm, sq.mm->sqmm, cu.m->cum, psi->bar...).

    IMPORTANT: `Standardized Description` is never used as a target or training
    signal here. That column is evaluation-only; using it would leak the answer.

INPUTS
    Raw description strings (and optionally the attribute columns for context).

OUTPUTS
    NormalizedText - dataclass with .text (canonical string) and .trace
                     (list of (pass_name, before, after) for the audit trail).

KEY FUNCTIONS
    normalize(text)                -> NormalizedText
    normalize_series(series)       -> pandas.Series[str]
    expand_abbreviations(text)     -> str
    canonicalize_units(text)       -> str
"""
