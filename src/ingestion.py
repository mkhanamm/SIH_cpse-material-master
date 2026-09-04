"""
Data ingestion and profiling (Spec section 4.1).

WHAT THIS FILE DOES
    Loads the CPSE material master workbook, splits it into a *pipeline* frame
    (what the matcher is allowed to see) and an *evaluation* frame (labels the
    matcher must never see), and computes profiling statistics for the dashboard.

    The split is enforced structurally, not by convention: `Standardized
    Description` and `GroundTruth_Group` are physically removed from the frame
    returned as pipeline input, so leakage cannot happen by accident.

INPUTS
    Path to `cpse_synthetic_dataset_with_cpse_mapping.xlsx` (sheet `All_Materials`).

OUTPUTS
    MaterialDataset  - dataclass holding .pipeline_df, .eval_df, .profile
    ProfileStats     - dataclass of row counts, null rates, distinct UOMs, etc.

KEY FUNCTIONS
    load_dataset(path)          -> MaterialDataset
    profile_dataframe(df)       -> ProfileStats
    ground_truth_pairs(eval_df) -> set[tuple[int, int]]
"""
