"""
Clustering into equivalence groups and confidence routing (Spec section 4.6).

WHAT THIS FILE DOES
    Turns scored pairs into material equivalence groups, then routes each group
    to an outcome tier.

    Clustering: a graph over materials, edges = pairs above the match threshold,
    equivalence groups = connected components. Because naive connected
    components can chain (A~B, B~C, A!~C) into one oversized cluster, a cohesion
    check splits any component whose internal average score falls below
    config.MIN_CLUSTER_COHESION.

    Confidence tiers (thresholds in config.py):
      HIGH   - auto-suggest a Common National Material Code.
      MEDIUM - queue for human review (this is the review_workflow input).
      LOW    - no match asserted.
      UNKNOWN- too many missing attributes to decide; explicitly flagged
               "insufficient data, needs manual review" rather than guessed.

INPUTS
    Scored/classified pair table.

OUTPUTS
    MatchResult - .clusters (list of MaterialCluster), .tier_counts
    MaterialCluster - members, mean/min internal score, tier, evidence refs

KEY FUNCTIONS
    build_graph(scored_pairs, threshold) -> networkx.Graph
    cluster(graph)                       -> list[MaterialCluster]
    assign_tiers(clusters)               -> list[MaterialCluster]
    evaluate(clusters, truth_pairs)      -> dict  (precision/recall/F1,
                                            within- vs cross-CPSE breakdown)
"""
