"""
Common National Material Code generation and mapping (Spec section 4.9).

WHAT THIS FILE DOES
    Assigns one stable national code per approved equivalence group and
    maintains the bidirectional mapping back to every contributing CPSE code,
    so nothing is ever lost in the harmonization.

    CNMC schema:  NM-<SS>-<CC>-<NNNNN>
        SS    sector digit-pair   (OG, PW, ST, MN, HE)
        CC    category short code (PIP, BRG, TRF, ...)
        NNNNN zero-padded sequence
    Codes are semantic enough for a procurement officer to read, but the
    sequence -- not the semantics -- is the identity: codes are never recycled
    and never rewritten when attributes change.

    Mapping table is the deliverable artifact:
        CNMC -> [(CPSE, CPSE Material Code, Legacy_Sector_Code), ...]
    with reverse lookup so a legacy code resolves to its national code.

INPUTS
    Approved MaterialClusters + the source dataframe.

OUTPUTS
    outputs/cnmc_mapping.csv - the national mapping table
    CNMCRegistry             - in-memory registry with forward/reverse lookup

KEY FUNCTIONS
    CNMCRegistry.assign(cluster)      -> str  (the new CNMC)
    CNMCRegistry.lookup(cnmc)         -> list[MemberRecord]
    CNMCRegistry.reverse_lookup(code) -> str | None
    CNMCRegistry.merge(a, b) / .split(cnmc, groups)
    export_mapping(registry, path)
"""
