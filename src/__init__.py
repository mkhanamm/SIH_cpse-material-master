"""
CPSE Unified Material Master — source package.

Pipeline order (each module does exactly one job):
    ingestion -> normalization -> attribute_extraction -> blocking ->
    similarity -> classifier -> matching_engine -> explanation ->
    review_workflow -> cnmc_generator -> governance

`config.py` holds every tunable constant. `app.py` (repo root) is UI wiring only.
"""

__version__ = "0.1.0"
