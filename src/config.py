"""
Single source of every tunable constant.

WHAT THIS FILE DOES
    Centralises thresholds, weights, paths and backend selection so behaviour
    can be inspected and adjusted in one place instead of hunting across
    modules. Nothing here contains logic -- only named values and their
    justification comments.

SECTIONS
    Paths | Column names | Semantic backend | Similarity weights |
    Attribute tolerances | Confidence thresholds | Blocking parameters |
    CNMC schema | Impact-estimate assumptions
"""
