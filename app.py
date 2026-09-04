"""
Streamlit demo application -- the single judged deliverable (Spec section 3.5).

WHAT THIS FILE DOES
    UI wiring only. Every computation is imported from `src/`; this file holds
    no business logic, so the pipeline can be run, tested and reviewed without
    Streamlit, and the app cannot silently diverge from the library.

    Six views:
      1. The Problem            - real cross-CPSE duplicate examples from the data
      2. Run Matching           - executes the pipeline, shows blocking speedup
      3. Review a Match         - per-attribute explanation for a chosen cluster
      4. Human Review           - approve/reject/edit + active-learning recalibration
      5. National Code Generated- CNMC assignment and the mapping table
      6. Dashboard              - duplicate rate, cross-CPSE count, savings estimate

RUN
    streamlit run app.py
"""
