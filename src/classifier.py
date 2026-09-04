"""
Trained duplicate classifier over similarity features (Spec section 4.5).

WHAT THIS FILE DOES
    Learns the pair -> duplicate decision instead of hand-tuning a threshold.
    For each candidate pair it builds a feature vector from the similarity
    channels (semantic, string, per-attribute agreement flags, missingness
    indicators), labels it from `GroundTruth_Group` (same group = duplicate),
    splits GROUP-WISE so that no ground-truth group appears in both train and
    test, fits a small model, and reports held-out precision/recall/F1.

    Group-wise splitting matters: a random pair-level split would put two pairs
    from the same cluster on both sides and inflate the score. The honest number
    is the one where the test clusters were never seen.

    Model is deliberately small and inspectable -- logistic regression by default
    (coefficients readable as evidence weights), gradient-boosted trees optional.

INPUTS
    Scored candidate pairs + evaluation labels.

OUTPUTS
    models/classifier.pkl  - fitted sklearn pipeline
    ClassifierReport       - precision/recall/F1 overall and split by
                             within-CPSE vs cross-CPSE, plus feature weights.

KEY FUNCTIONS
    build_features(scored_pairs)          -> (X, feature_names)
    label_pairs(pairs, eval_df)           -> ndarray
    train(X, y, groups)                   -> (model, ClassifierReport)
    save_model(model) / load_model()
"""
