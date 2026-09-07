"""
Streamlit demo application -- the single judged deliverable (Spec section 3.5).

WHAT THIS FILE DOES
    UI wiring only. Every computation is imported from `src/`; this file holds
    no business logic, so the pipeline can be run, tested and reviewed without
    Streamlit, and the app cannot silently diverge from the library.

    Seven views:
      1. Load Data              - pick the demo dataset or upload and map your own
      2. The Problem            - real cross-CPSE duplicate examples from the data
      3. Run Matching           - executes the pipeline, shows blocking speedup
      4. Review a Match         - per-attribute explanation for a chosen cluster
      5. Human Review           - approve/reject/edit + active-learning recalibration
      6. National Code Generated- CNMC assignment and the mapping table
      7. Dashboard              - duplicate rate, cross-CPSE count, savings estimate

RUN
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import attribute_extraction, blocking, classifier, config, cnmc_generator
from src import data_loading
from src import explanation as explain
from src import governance, ingestion, matching_engine, review_workflow, similarity
from src.normalization import normalize_series

st.set_page_config(
    page_title="National Unified Material Master (CPSE)",
    page_icon=None,
    layout="wide",
)

VIEWS = [
    "1. Load Data",
    "2. The Problem",
    "3. Run Matching",
    "4. Review a Match",
    "5. Human Review",
    "6. National Code Generated",
    "7. Dashboard",
]

SYSTEM_ACTOR = "auto-matcher"


# ---------------------------------------------------------------------------
# Cached pipeline stages
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Normalizing and extracting attributes...")
def load_everything(_dataset: ingestion.MaterialDataset, dataset_token: str):
    """Run normalization and attribute extraction once per loaded dataset.

    Cached on ``dataset_token`` because these are deterministic and take a
    few seconds; a judge clicking between tabs should not re-pay that cost.
    ``_dataset`` is excluded from the cache key (Streamlit's ``_``-prefix
    convention) since a DataFrame-holding object is neither cheap to hash
    nor a meaningful identity check -- ``dataset_token`` (set by
    :func:`_set_active_dataset` whenever "Load Data" loads a new dataset) is
    what actually determines whether this needs to re-run.

    Args:
        _dataset: The active dataset, from ``st.session_state.dataset``.
        dataset_token: Identifies which dataset this is, for cache-busting.

    Returns:
        ``(dataset, joined_frame, normalized_text, extracted_attributes)``.
    """
    dataset = _dataset
    extracted = attribute_extraction.extract_frame(dataset.pipeline_df)
    joined = dataset.pipeline_df.join(extracted)
    text = normalize_series(joined[config.INPUT_TEXT_COLUMN])
    return dataset, joined, text, extracted


@st.cache_resource(show_spinner="Blocking, scoring and clustering...")
def run_pipeline(_dataset: ingestion.MaterialDataset, dataset_token: str):
    """Execute blocking, similarity, classification and clustering.

    Two paths, chosen by ``dataset.has_labels``:

    LABELLED (``GroundTruth_Group`` present, e.g. the synthetic dataset) --
    trains the classifier, tunes the edge threshold against ground truth, and
    calibrates tiers and accuracy metrics from it.

    UNLABELLED (real CPSE uploads) -- there is no answer key, so training,
    edge-threshold tuning and accuracy scoring all have nothing to run
    against. This path loads the shipped ``models/classifier.pkl``
    (``classifier.load_model_or_none``) and reuses its ``match_probability``
    with the fixed cut-offs in ``classifier.pretrained_tiers`` -- calibrated
    on the demo dataset, but the model's features are dataset-agnostic so
    they transfer without retraining. Clustering, explanations, CNMC
    generation, review and the audit trail are unaffected -- only the
    accuracy metrics are unavailable. If no model artifact exists at all,
    this degrades further to the hand-fused score
    (``classifier.fallback_tiers``), which measured far worse (see
    ``config.py``) -- ``degraded_fallback`` in the return value flags this so
    the UI can warn plainly.

    The pre-trained model's features are dataset-agnostic, but SBERT and
    tfidf_svd (``config.SEMANTIC_BACKEND = "auto"``) produce different score
    distributions, and ``"auto"`` can resolve differently on the machine that
    trained ``models/classifier.pkl`` versus the one running it now (e.g. no
    sentence-transformers installed). When that happens,
    ``classifier.is_backend_mismatch`` flags it and ``backend_mismatch`` in
    the return value carries ``(trained_backend, actual_backend)`` so the UI
    can warn that the pre-trained cut-offs may no longer apply.

    Returns:
        A dict with every artifact the views need: blocking stats, scored
        pairs, classifier report (None if unlabelled), tier thresholds, match
        result, the edge-threshold sweep (None if unlabelled), accuracy
        metrics (None if unlabelled), ``has_labels``, ``degraded_fallback``
        and ``backend_mismatch`` (None unless a genuine mismatch was found).
    """
    dataset, joined, text, extracted = load_everything(_dataset, dataset_token)
    candidates = blocking.candidate_pairs(joined)
    scored = similarity.score_pairs(joined, candidates.pairs, text)

    if dataset.has_labels:
        truth = ingestion.ground_truth_pairs(dataset.eval_df)
        blocking_stats = blocking.evaluate_blocking(candidates, truth)

        model, report = classifier.train(scored, dataset.eval_df)
        scored["match_probability"] = classifier.predict_proba(model, scored)

        _, _, test_mask, _ = classifier.group_disjoint_split(scored, dataset.eval_df)
        test_records = set(scored.loc[test_mask, "idx_a"]) | set(
            scored.loc[test_mask, "idx_b"]
        )
        tuning_records = set(joined.index) - test_records

        edge_threshold, sweep = matching_engine.tune_edge_threshold(
            scored, joined, truth, tuning_records, extracted["n_known_attributes"]
        )
        tiers = matching_engine.calibrate_tiers_from_sweep(sweep, edge_threshold)
        result = matching_engine.run_matching(
            scored,
            joined,
            tiers,
            edge_threshold=edge_threshold,
            attribute_counts=extracted["n_known_attributes"],
        )
        metrics = matching_engine.evaluate(
            result,
            truth,
            joined,
            tiers=(
                matching_engine.HIGH, matching_engine.MEDIUM, matching_engine.UNKNOWN,
            ),
        )
        degraded_fallback = False
        backend_mismatch = None
    else:
        truth = None
        blocking_stats = candidates.stats
        report = None
        backend_mismatch = None

        model = classifier.load_model_or_none()
        degraded_fallback = model is None
        if model is not None:
            scored["match_probability"] = classifier.predict_proba(model, scored)
            tiers = classifier.pretrained_tiers()

            trained_backend = classifier.load_model_backend()
            actual_backend = scored.attrs.get("semantic_backend")
            if classifier.is_backend_mismatch(trained_backend, actual_backend):
                backend_mismatch = (trained_backend, actual_backend)
        else:
            scored["match_probability"] = scored["fused"]
            tiers = classifier.fallback_tiers()

        edge_threshold = tiers.medium
        sweep = None
        result = matching_engine.run_matching(
            scored,
            joined,
            tiers,
            edge_threshold=edge_threshold,
            attribute_counts=extracted["n_known_attributes"],
        )
        metrics = None

    return {
        "has_labels": dataset.has_labels,
        "degraded_fallback": degraded_fallback,
        "backend_mismatch": backend_mismatch,
        "blocking": blocking_stats,
        "scored": scored,
        "report": report,
        "tiers": tiers,
        "edge_threshold": edge_threshold,
        "sweep": sweep,
        "result": result,
        "metrics": metrics,
        "truth": truth,
        "attribute_lookup": {
            i: r.to_dict()
            for i, r in extracted[attribute_extraction.ALL_ATTRIBUTES].iterrows()
        },
    }


def session_state_defaults() -> None:
    """Initialise mutable session state used across views."""
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("dataset_token", None)
    st.session_state.setdefault("decisions", [])
    st.session_state.setdefault("registry", cnmc_generator.CNMCRegistry())
    st.session_state.setdefault("audit", governance.AuditLog(load=False))
    st.session_state.setdefault("pipeline_run", False)
    st.session_state.setdefault("pending_view", None)


def _apply_pending_view_change() -> None:
    """Consume a requested view jump, if any, before the view radio renders.

    Streamlit forbids writing to a widget-bound session-state key (like the
    view radio's) after that widget has already been instantiated in the
    current script run -- so a button inside a view function can't jump
    straight to another view by setting ``view_choice`` itself. Instead it
    sets ``pending_view`` (a plain, non-widget key) and calls ``st.rerun()``;
    this, called at the very top of ``main()`` on the resulting rerun,
    applies that request before the radio widget is created.
    """
    if st.session_state.pending_view is not None:
        st.session_state.view_choice = st.session_state.pending_view
        st.session_state.pending_view = None


def _set_active_dataset(dataset: ingestion.MaterialDataset, token: str) -> None:
    """Replace the active dataset and reset everything that assumed the old one.

    Args:
        dataset: The newly loaded dataset (demo or mapped upload).
        token: Cache-busting identity for :func:`load_everything` /
            :func:`run_pipeline` -- must change whenever the dataset changes.
    """
    st.session_state.dataset = dataset
    st.session_state.dataset_token = token
    st.session_state.pipeline_run = False
    st.session_state.decisions = []
    st.session_state.registry = cnmc_generator.CNMCRegistry()
    st.session_state.audit = governance.AuditLog(load=False)


def _require_dataset() -> bool:
    """Show a warning and report failure if no dataset has been loaded yet.

    Returns:
        True if a dataset is loaded and the caller may proceed.
    """
    if st.session_state.dataset is None:
        st.warning("Load a dataset first, on the '1. Load Data' view.")
        return False
    return True


def _active_dataset_args() -> tuple[ingestion.MaterialDataset, str]:
    """Arguments for ``load_everything``/``run_pipeline``: (dataset, token).

    Callers must check :func:`_require_dataset` first -- this does not
    itself guard against ``st.session_state.dataset`` being None.

    Returns:
        The active dataset and its cache-busting token.
    """
    return st.session_state.dataset, st.session_state.dataset_token


def _format_seconds(seconds: float) -> str:
    """Render an estimated duration for the "Load Data" view.

    Args:
        seconds: Estimated seconds.

    Returns:
        ``"~<n>s"`` under 90 seconds, otherwise ``"~<n> min"``.
    """
    if seconds < 90:
        return f"~{seconds:.0f}s"
    return f"~{seconds / 60:.1f} min"


def _reset_session() -> None:
    """Clear every cached artifact and mutable session entry, for the
    sidebar's "Start over / load a different dataset" button.

    Clears ``st.cache_resource`` (so ``load_everything``/``run_pipeline``
    recompute from scratch for whatever is loaded next), drops the active
    dataset so "Load Data" shows the source picker rather than an
    already-loaded dataset, and resets every other piece of mutable session
    state (decisions, registry, audit log, pipeline_run) -- the same reset
    :func:`_set_active_dataset` does when swapping datasets, minus setting a
    new one. Does not itself rerun the app; call ``st.rerun()`` after.
    """
    st.cache_resource.clear()
    st.session_state.dataset = None
    st.session_state.dataset_token = None
    st.session_state.pipeline_run = False
    st.session_state.decisions = []
    st.session_state.registry = cnmc_generator.CNMCRegistry()
    st.session_state.audit = governance.AuditLog(load=False)
    st.session_state.pop("data_source_choice", None)
    st.session_state.pending_view = VIEWS[0]


def _pipeline_not_run_yet(what_this_view_shows: str) -> None:
    """Friendly placeholder for a view that needs a completed pipeline run.

    Replaces a bare "run the pipeline first" warning with a description of
    what will actually appear here, plus a direct way to get there, instead
    of just naming the view the user has to go find themselves.

    Args:
        what_this_view_shows: One sentence, present tense, describing what
            this view shows once the pipeline has run.
    """
    st.info(f"Once you've run the pipeline, this view will show {what_this_view_shows}")
    if st.button("Go to Run Matching", key="goto_run_matching"):
        st.session_state.pending_view = VIEWS[2]
        st.rerun()


# ---------------------------------------------------------------------------
# View 1 -- Load Data
# ---------------------------------------------------------------------------
def view_load_data() -> None:
    """Pick the demo dataset, or upload and map a real CPSE export."""
    st.header("Load a material master")
    st.write(
        "Everything downstream reads from whatever is loaded here: the "
        "bundled SIH demo dataset, or your own export mapped onto the same "
        "columns."
    )

    source = st.radio(
        "Data source",
        ["Use the SIH demo dataset", "Upload your own"],
        key="data_source_choice",
    )

    if source == "Use the SIH demo dataset":
        if st.session_state.dataset_token != "demo":
            with st.spinner("Loading the demo dataset..."):
                _set_active_dataset(ingestion.load_dataset(), "demo")
        dataset = st.session_state.dataset
        st.success(
            f"Demo dataset loaded: {dataset.profile.n_rows:,} records, "
            f"{dataset.profile.n_cpses} CPSEs, {dataset.profile.n_categories} "
            "material categories. Continue to '2. The Problem' or "
            "'3. Run Matching'."
        )
        return

    uploaded = st.file_uploader("Upload a material master", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("Choose a .xlsx or .csv file to continue.")
        return

    try:
        raw_df = data_loading.read_upload(uploaded, uploaded.name)
    except (data_loading.UnsupportedFileType, ValueError) as error:
        st.error(f"Could not read `{uploaded.name}`: {error}")
        return

    st.write(f"**{len(raw_df):,} rows, {len(raw_df.columns)} columns** in `{uploaded.name}`.")
    st.subheader("Preview (first 10 rows)")
    st.dataframe(raw_df.head(10), width="stretch")

    st.subheader("Map your columns")
    st.caption(
        "Required: CPSE, CPSE Material Code, Material Category, Raw "
        "Description. Optional: Sector and the five attribute columns -- "
        "leave any of these as '-- none --' if your file doesn't have them. "
        "Each dropdown is pre-filled with a best guess -- from the header "
        "name where it matches, otherwise from the shape of the column's "
        "values -- so check them rather than assume they're right."
    )
    st.caption(
        "CPSE is the company that owns the code, CPSE Material Code is "
        "that company's own part number, Material Category is the type of "
        "item, and Raw Description is the free-text description the "
        "system reads. Sector is optional but improves the national code "
        "format."
    )
    suggested = data_loading.infer_column_mapping(raw_df)
    options = ["-- none --", *raw_df.columns]
    hints = {column: data_loading.column_hint(raw_df[column]) for column in raw_df.columns}

    def _option_label(option: str) -> str:
        hint = hints.get(option)
        return f"{option}  ({hint})" if hint else option

    def _mapping_selectbox(target: str) -> str | None:
        default = suggested.get(target)
        index = options.index(default) if default in options else 0
        choice = st.selectbox(
            target, options, index=index, key=f"col_map_{target}",
            format_func=_option_label,
        )
        return None if choice == "-- none --" else choice

    st.write("**Required**")
    mapping = {target: _mapping_selectbox(target) for target in data_loading.REQUIRED_COLUMNS}
    st.write("**Optional**")
    mapping.update(
        {target: _mapping_selectbox(target) for target in data_loading.OPTIONAL_COLUMNS}
    )

    st.caption("Example row with this mapping:")
    preview_index = 0
    if len(raw_df) > 1:
        preview_index = st.number_input(
            "Preview row", min_value=0, max_value=len(raw_df) - 1, value=0, step=1
        )
    st.code(
        data_loading.preview_mapped_row(raw_df, mapping, int(preview_index)),
        language=None,
    )

    missing_required = [t for t in data_loading.REQUIRED_COLUMNS if not mapping.get(t)]
    if missing_required:
        st.warning(f"Map the required column(s) first: {', '.join(missing_required)}.")
        return

    collisions = data_loading.duplicate_required_sources(mapping)
    if collisions:
        for source, fields in collisions.items():
            st.error(
                f"Column **{source}** is mapped to more than one required "
                f"field ({', '.join(fields)}). Each required field needs its "
                "own column -- fix the mapping above before loading."
            )
        return

    n_rows = len(raw_df)
    estimate = data_loading.estimate_runtime_seconds(n_rows)
    st.subheader("Before you run it")
    columns = st.columns(2)
    columns[0].metric("Rows", f"{n_rows:,}")
    columns[1].metric("Estimated pipeline runtime", _format_seconds(estimate))

    n_sample = n_rows
    if n_rows > config.LARGE_DATASET_WARNING_ROWS:
        st.warning(
            f"{n_rows:,} rows is large. Comparisons -- and therefore "
            "runtime -- grow faster than linearly with record count "
            f"(measured: {config.RUNTIME_BASELINE_ROWS:,} records in about "
            f"{config.RUNTIME_BASELINE_SECONDS}s), so a full run here is "
            f"estimated at {_format_seconds(estimate)}. Sample a subset for "
            "a faster demo."
        )
        if st.checkbox("Sample a subset of rows", value=True):
            n_sample = st.slider(
                "Sample size",
                min_value=1_000,
                max_value=n_rows,
                value=min(config.LARGE_DATASET_WARNING_ROWS, n_rows),
                step=1_000,
            )
            st.caption(
                f"Estimated runtime at {n_sample:,} rows: "
                f"{_format_seconds(data_loading.estimate_runtime_seconds(n_sample))}."
            )

    if st.button("Load this dataset", type="primary"):
        mapped_df = data_loading.apply_column_mapping(raw_df, mapping)
        dataset = data_loading.build_dataset_from_mapped(mapped_df, uploaded.name)
        if n_sample < len(dataset.pipeline_df):
            dataset = ingestion.sample_dataset(dataset, n_sample)
        _set_active_dataset(dataset, f"upload:{uploaded.name}:{n_sample}")
        st.success(
            f"Loaded {dataset.profile.n_rows:,} records from `{uploaded.name}`. "
            "Continue to '2. The Problem' or '3. Run Matching'."
        )


# ---------------------------------------------------------------------------
# View 2 -- The Problem
# ---------------------------------------------------------------------------
def view_problem() -> None:
    """Show real cross-CPSE duplicates so the pain precedes the solution."""
    if not _require_dataset():
        return
    dataset, joined, _, _ = load_everything(*_active_dataset_args())

    st.header("The same material, coded differently by every enterprise")
    st.write(
        "Each Central Public Sector Enterprise maintains its own material "
        "master. Nothing forces two of them to describe the same physical item "
        "the same way, so the same pipe, bearing or valve carries a different "
        "code and a different description at every enterprise that buys it. "
        "Procurement cannot aggregate demand it cannot see."
    )

    st.subheader("How the loaded data is structured")
    st.caption(
        "Every record belongs to one CPSE and one sector -- a sector groups "
        "related CPSEs (e.g. every oil & gas company)."
    )
    structure_columns = st.columns(2)
    with structure_columns[0]:
        st.write("**Records by sector**")
        st.bar_chart(
            pd.Series(dataset.profile.rows_per_sector).sort_values(ascending=False)
        )
    with structure_columns[1]:
        st.write("**Records by CPSE**")
        st.bar_chart(
            pd.Series(dataset.profile.rows_per_cpse).sort_values(ascending=False)
        )

    if not dataset.has_labels:
        st.info(
            "The worked examples below use the dataset's ground-truth "
            "labels (`GroundTruth_Group`). The loaded dataset has none, so "
            "they are unavailable -- every other view (matching, review, "
            "national code generation, dashboard) still runs normally."
        )
        return

    summary = ingestion.ground_truth_summary(dataset.eval_df, dataset.pipeline_df)

    columns = st.columns(4)
    columns[0].metric("Material records", f"{summary['total_groups']:,} groups")
    columns[1].metric("Duplicate groups", f"{summary['multi_member_groups']:,}")
    columns[2].metric("Cross-CPSE groups", f"{summary['cross_cpse_groups']:,}")
    columns[3].metric("Cross-CPSE pairs", f"{summary['cross_cpse_pairs']:,}")

    st.subheader("Real examples from the loaded data")
    st.caption(
        "Every description below is textually different from the others in its "
        "group. Exact string matching finds none of these. Sector and "
        "Material Category are shown so it's clear how materials are "
        "categorised."
    )

    examples = ingestion.find_cross_cpse_examples(dataset, limit=None)
    if not examples:
        st.info("No cross-CPSE duplicate examples with distinct text were found.")
        return

    sectors = sorted({example["Sector"].iat[0] for example in examples})
    categories = sorted({example["Material Category"].iat[0] for example in examples})

    filter_columns = st.columns(2)
    sector_choice = filter_columns[0].selectbox("Filter by sector", ["All"] + sectors)
    category_choice = filter_columns[1].selectbox(
        "Filter by category", ["All"] + categories
    )

    filtered = [
        example
        for example in examples
        if sector_choice in ("All", example["Sector"].iat[0])
        and category_choice in ("All", example["Material Category"].iat[0])
    ]

    if not filtered:
        st.info(
            f"No examples match Sector={sector_choice!r}, "
            f"Category={category_choice!r}. Try a different combination."
        )

    for example in filtered[:6]:
        n_codes = len(example)
        n_companies = example["CPSE"].nunique()
        st.write(
            f"**The same material, coded {n_codes} different ways by "
            f"{n_companies} companies**"
        )
        st.dataframe(example, width="stretch", hide_index=True)

    st.info(
        "Exact matching on normalized text already recovers 74.2% of true "
        "duplicate pairs. The remaining quarter -- where wording, abbreviation "
        "and units all differ -- is what the matching engine has to earn."
    )


# ---------------------------------------------------------------------------
# View 3 -- Run Matching
# ---------------------------------------------------------------------------
def view_run_matching() -> None:
    """Execute the pipeline and show cost and quality side by side."""
    st.header("Run the matching pipeline")
    if not _require_dataset():
        return
    st.write(
        "Ingestion to clustering, on the full loaded master. The blocking "
        "figures are the reason this is feasible at national scale."
    )

    if st.button("Run pipeline", type="primary") or st.session_state.pipeline_run:
        st.session_state.pipeline_run = True
        artifacts = run_pipeline(*_active_dataset_args())
        has_labels = artifacts["has_labels"]
        stats = artifacts["blocking"]

        st.subheader("Blocking")
        columns = st.columns(4)
        columns[0].metric("All-pairs comparisons", f"{stats.naive_comparisons:,}")
        columns[1].metric("After blocking", f"{stats.blocked_comparisons:,}")
        columns[2].metric("Reduction", f"{stats.reduction_factor:.1f}x")
        if has_labels:
            columns[3].metric("Recall ceiling", f"{stats.recall_ceiling:.2%}")
            st.caption(
                "A speedup figure without its recall cost is not a "
                "measurement. This scheme loses nothing: the category-token "
                "key recovers the pairs that exact-category blocking cannot "
                "reach."
            )
        else:
            columns[3].metric("Recall ceiling", "N/A")
            st.caption(
                "No GroundTruth_Group column: recall against ground truth is "
                "unavailable for this dataset."
            )

        st.subheader("Classifier, held out on unseen clusters")
        if has_labels:
            for line in artifacts["report"].summary_lines():
                st.text(line)
        elif not artifacts["degraded_fallback"]:
            if artifacts["backend_mismatch"]:
                trained_backend, actual_backend = artifacts["backend_mismatch"]
                st.error(
                    f"**Backend mismatch**: `models/classifier.pkl` was "
                    f"trained on `{trained_backend}` features, but this run "
                    f"scored pairs with `{actual_backend}` -- "
                    "`config.SEMANTIC_BACKEND = \"auto\"` resolved "
                    "differently on this machine (for example, "
                    "sentence-transformers isn't installed, or there's no "
                    "network access). The two backends produce different "
                    "score distributions, so `match_probability` and the "
                    "PRETRAINED_EDGE_THRESHOLD/PRETRAINED_HIGH_THRESHOLD "
                    f"cut-offs -- calibrated for `{trained_backend}` -- may "
                    "be miscalibrated for this run. Regenerate the model on "
                    "this machine with `python -m src.classifier`, or match "
                    "its environment (install sentence-transformers / "
                    "restore network access) to clear this warning."
                )
            st.info(
                "No GroundTruth_Group column: skipping classifier training. "
                "Using the shipped pre-trained classifier "
                "(`models/classifier.pkl`) instead -- its features "
                "(similarity scores, attribute agreement) describe the pair, "
                "not the dataset, so it transfers without retraining. The "
                "edge threshold (0.55) and HIGH tier (0.85) were calibrated "
                "on the demo dataset and may need adjusting for a "
                "materially different catalogue."
            )
        else:
            st.warning(
                "No GroundTruth_Group column AND no `models/classifier.pkl` "
                "found: falling back to the hand-fused similarity score. "
                "Measured on the demo dataset with labels stripped, this "
                "fallback reaches only ~0.26 F1 at best (precision as low "
                "as 0.024 at threshold 0.65) against 0.897 for the "
                "trained-classifier path -- **treat these results as "
                "indicative only.**"
            )

        st.subheader("Confidence routing")
        for line in artifacts["tiers"].summary_lines():
            st.text(line)
        st.write(
            pd.DataFrame(
                [artifacts["result"].tier_counts]
            ).rename(index={0: "clusters"})
        )

        st.subheader("Cluster-level accuracy against ground truth")
        if has_labels:
            metrics = artifacts["metrics"]
            table = pd.DataFrame(
                [
                    {
                        "scope": scope.replace("_", "-"),
                        "precision": round(metrics[f"{scope}_precision"], 3),
                        "recall": round(metrics[f"{scope}_recall"], 3),
                        "f1": round(metrics[f"{scope}_f1"], 3),
                        "true duplicates": metrics[f"{scope}_actual"],
                    }
                    for scope in ("overall", "cross_cpse", "within_cpse")
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info(
                "No ground-truth labels in this dataset, so accuracy metrics "
                "are unavailable. Clustering, explanations, CNMC generation, "
                "review and the audit trail all still ran on the tiers shown "
                "above."
            )

        with st.expander("Edge-threshold sweep (why the cut-off is what it is)"):
            if has_labels:
                st.caption(
                    "The pair-optimal threshold is not the cluster-optimal "
                    "one. A cluster asserts equivalence between every pair "
                    "of its members, so one bad edge joining two correct "
                    "clusters of five manufactures 25 false pairs."
                )
                st.dataframe(artifacts["sweep"], width="stretch", hide_index=True)
            else:
                st.caption(
                    "The sweep is measured against ground-truth pairs, which "
                    "this dataset does not have. The edge threshold instead "
                    "uses the fixed cut-off shown above -- calibrated on the "
                    "demo dataset ("
                    + (
                        "pre-trained classifier probability"
                        if not artifacts["degraded_fallback"]
                        else "hand-fused score, degraded fallback"
                    )
                    + ")."
                )


# ---------------------------------------------------------------------------
# View 4 -- Review a Match
# ---------------------------------------------------------------------------
def view_review_match() -> None:
    """Show a full per-attribute explanation for one chosen cluster."""
    st.header("Why did the system propose this match?")
    if not st.session_state.pipeline_run:
        _pipeline_not_run_yet(
            "a per-attribute explanation for one proposed cluster you "
            "choose, alongside its confidence tier and internal scores."
        )
        return

    _, joined, _, _ = load_everything(*_active_dataset_args())
    artifacts = run_pipeline(*_active_dataset_args())
    clusters = [c for c in artifacts["result"].clusters if c.is_cross_cpse]

    if not clusters:
        st.info("No cross-CPSE clusters were produced.")
        return

    labels = {
        f"Cluster {c.cluster_id} - {c.size} records, {', '.join(c.cpses)} "
        f"[{c.tier}]": c
        for c in clusters[:200]
    }
    chosen = labels[st.selectbox("Choose a proposed match", list(labels))]

    material_explanation = explain.explain_cluster(
        chosen,
        joined,
        artifacts["scored"],
        artifacts["attribute_lookup"],
        hard_conflict_attributes=tuple(config.EXACT_MATCH_ATTRIBUTES),
    )

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Evidence")
        st.code(material_explanation.as_text(), language=None)
    with right:
        st.subheader("Scores")
        st.metric("Weakest internal link", f"{chosen.min_score:.3f}")
        st.metric("Mean internal score", f"{chosen.mean_score:.3f}")
        st.metric("Confidence tier", chosen.tier)
        st.caption(
            f"Semantic backend: {artifacts['scored'].attrs.get('semantic_backend')}"
        )

    st.caption(
        "Explanations are generated from a deterministic template, not a live "
        "language model. That is a demo-reliability choice: no API key, no "
        "network dependency, and byte-identical output on every run. "
        "`explanation.render_explanation` is the single seam where a live LLM "
        "would be substituted for richer prose."
    )


# ---------------------------------------------------------------------------
# View 5 -- Human Review
# ---------------------------------------------------------------------------
def view_human_review() -> None:
    """Approve, reject or edit queued clusters and show the feedback effect."""
    st.header("Human review queue")
    if not st.session_state.pipeline_run:
        _pipeline_not_run_yet(
            "the medium-confidence review queue, where you can approve, "
            "reject or edit proposed matches and see the effect of "
            "feedback on accuracy."
        )
        return

    dataset, joined, _, _ = load_everything(*_active_dataset_args())
    artifacts = run_pipeline(*_active_dataset_args())
    queue = review_workflow.build_queue(
        artifacts["result"].clusters, artifacts["tiers"].medium
    )

    st.caption(
        f"{len(queue)} clusters queued, ordered by uncertainty rather than by "
        "score. The cases nearest the decision boundary teach the model most, "
        "so they are shown first."
    )

    reviewer = st.text_input("Reviewer identity", value="reviewer.demo")
    item = queue[st.number_input(
        "Queue position", min_value=0, max_value=max(0, len(queue) - 1), value=0
    )] if queue else None

    if item is None:
        st.info("Queue is empty.")
        return

    st.subheader(f"Cluster {item.cluster_id} - {item.tier}")
    st.write(f"Score {item.score:.3f}, uncertainty {item.uncertainty:.3f}")
    st.write(item.reason)
    st.dataframe(
        joined.loc[
            item.members,
            ["CPSE", "CPSE Material Code", "Material Category", "Raw Description"],
        ],
        width="stretch",
    )

    comment = st.text_input("Comment", value="")
    approve, reject, edit = st.columns(3)

    if approve.button("Approve"):
        st.session_state.decisions.append(
            review_workflow.record_decision(item, "approve", reviewer, comment)
        )
        st.success(f"Approved by {reviewer}.")
    if reject.button("Reject"):
        st.session_state.decisions.append(
            review_workflow.record_decision(item, "reject", reviewer, comment)
        )
        st.success(f"Rejected by {reviewer}.")

    keep = st.multiselect(
        "Corrected membership (for Edit)", item.members, default=item.members
    )
    if edit.button("Edit") and len(keep) >= 2:
        st.session_state.decisions.append(
            review_workflow.record_decision(
                item, "edit", reviewer, comment, corrected_members=keep
            )
        )
        st.success(f"Corrected to {len(keep)} records by {reviewer}.")

    st.divider()
    st.subheader("Active learning: does review actually help?")
    if not artifacts["has_labels"]:
        st.info(
            "This demo simulates a reviewer from ground-truth labels, which "
            "this dataset does not have. Approve/reject/edit above still "
            "work and are logged to the audit trail normally; only this "
            "simulated before/after comparison is unavailable."
        )
        return

    st.caption(
        "Simulated decisions are derived from ground truth so the loop can be "
        "demonstrated in a live session. They stand in for a reviewer; they are "
        "not evidence about how real reviewers behave."
    )

    n_simulated = st.slider("Simulated decisions", 10, 120, 60, step=10)
    if st.button("Run active-learning comparison"):
        simulated = review_workflow.simulate_reviews(
            queue, dataset.eval_df, n=n_simulated, log_path=config.REVIEW_LOG_PATH
        )
        applied = review_workflow.apply_decisions(
            artifacts["result"].clusters, simulated
        )
        after_result = matching_engine.MatchResult(
            applied, {}, artifacts["edge_threshold"], len(joined),
            sum(c.size for c in applied),
        )
        tiers_counted = (
            matching_engine.HIGH, matching_engine.MEDIUM, matching_engine.UNKNOWN
        )
        after = matching_engine.evaluate(
            after_result, artifacts["truth"], joined, tiers=tiers_counted
        )
        before = artifacts["metrics"]

        report = review_workflow.measure_improvement(
            {k.replace("overall_", ""): v for k, v in before.items()
             if k.startswith("overall_")},
            {k.replace("overall_", ""): v for k, v in after.items()
             if k.startswith("overall_")},
            len(simulated),
            len(review_workflow.decisions_to_labels(simulated)),
            artifacts["tiers"].medium,
            review_workflow.recalibrate_threshold(
                artifacts["scored"], simulated, artifacts["tiers"].medium
            ),
        )
        for line in report.summary_lines():
            st.text(line)


# ---------------------------------------------------------------------------
# View 6 -- National Code Generated
# ---------------------------------------------------------------------------
def view_national_code() -> None:
    """Assign national codes to approved clusters and show the mapping."""
    st.header("Common National Material Code")
    if not st.session_state.pipeline_run:
        _pipeline_not_run_yet(
            "assignment of national codes to auto-approved clusters, the "
            "resulting mapping table, legacy-code lookup and the append-only "
            "audit trail."
        )
        return

    _, joined, _, _ = load_everything(*_active_dataset_args())
    artifacts = run_pipeline(*_active_dataset_args())
    registry = st.session_state.registry
    audit = st.session_state.audit

    st.write(
        "Each approved equivalence group receives one national code. Every "
        "contributing CPSE keeps its own code unchanged -- the mapping is "
        "additive, so an enterprise can keep transacting on its existing code "
        "while procurement finally sees one material."
    )

    if st.button("Assign codes to auto-approved clusters", type="primary"):
        assigned = 0
        for material_cluster in artifacts["result"].clusters:
            if material_cluster.tier != matching_engine.HIGH:
                continue
            try:
                entry = registry.assign(
                    material_cluster, joined, created_by=SYSTEM_ACTOR
                )
            except ValueError:
                continue
            reason = explain.explain_cluster(
                material_cluster,
                joined,
                artifacts["scored"],
                artifacts["attribute_lookup"],
                hard_conflict_attributes=tuple(config.EXACT_MATCH_ATTRIBUTES),
            )
            audit.record(
                "create",
                SYSTEM_ACTOR,
                entry.cnmc,
                after={"members": [m.cpse_material_code for m in entry.members]},
                reason=reason.as_dict(),
            )
            assigned += 1
        st.success(f"{assigned} national codes issued and logged.")

    stats = registry.stats()
    if stats["codes_issued"]:
        columns = st.columns(4)
        columns[0].metric("Codes issued", f"{stats['active_codes']:,}")
        columns[1].metric("Cross-CPSE codes", f"{stats['cross_cpse_codes']:,}")
        columns[2].metric("Records mapped", f"{stats['records_mapped']:,}")
        columns[3].metric("Codes eliminated", f"{stats['codes_eliminated']:,}")

        st.subheader("Mapping table")
        mapping = registry.to_frame()
        st.dataframe(mapping.head(200), width="stretch", hide_index=True)
        st.download_button(
            "Export mapping (CSV round-trip for ERP load)",
            mapping.to_csv(index=False).encode("utf-8"),
            file_name="cnmc_mapping.csv",
            mime="text/csv",
        )

        st.subheader("Legacy code lookup")
        code = st.text_input("CPSE material code", value="")
        cpse = st.text_input("CPSE", value="")
        if code and cpse:
            resolved = registry.reverse_lookup(cpse.strip(), code.strip())
            st.write(f"National code: **{resolved}**" if resolved else "Not mapped.")

        st.subheader("Audit trail")
        st.caption(
            "Append-only. Rollback appends a compensating event rather than "
            "deleting history, so the record that a bad merge happened survives "
            "the correction."
        )
        st.dataframe(audit.to_frame().head(50), width="stretch", hide_index=True)

        integrity = audit.integrity_check()
        st.write(
            f"Integrity check: {'passed' if integrity['ok'] else 'FAILED'} "
            f"({integrity['n_events']} events, {integrity['n_rollbacks']} rollbacks)"
        )

        event_id = st.number_input(
            "Roll back event id", min_value=0,
            max_value=max(0, len(audit.events) - 1), value=0,
        )
        rollback_actor = st.text_input("Authorising actor", value="auditor.demo")
        if st.button("Roll back"):
            try:
                event = audit.rollback(
                    int(event_id), rollback_actor, "Reversed from demo UI."
                )
                st.success(
                    f"Event {event_id} reversed by compensating event "
                    f"{event.event_id}. Original event retained."
                )
            except (IndexError, ValueError) as error:
                st.error(str(error))


# ---------------------------------------------------------------------------
# View 7 -- Dashboard
# ---------------------------------------------------------------------------
def view_dashboard() -> None:
    """Programme-level totals, accuracy and a clearly-caveated savings estimate."""
    st.header("Programme dashboard")
    if not st.session_state.pipeline_run:
        _pipeline_not_run_yet(
            "programme-level totals, accuracy against ground truth (when "
            "available) and an estimated procurement-savings figure."
        )
        return

    dataset, joined, _, _ = load_everything(*_active_dataset_args())
    artifacts = run_pipeline(*_active_dataset_args())
    result = artifacts["result"]
    metrics = artifacts["metrics"]

    columns = st.columns(4)
    columns[0].metric("Materials scanned", f"{len(joined):,}")
    columns[1].metric(
        "Records in duplicate clusters", f"{result.n_clustered_records:,}"
    )
    columns[2].metric(
        "Duplicate rate", f"{result.n_clustered_records / len(joined):.1%}"
    )
    columns[3].metric(
        "Cross-CPSE clusters",
        f"{sum(1 for c in result.clusters if c.is_cross_cpse):,}",
    )

    st.subheader("Validation against known ground truth")
    if artifacts["has_labels"]:
        summary = ingestion.ground_truth_summary(dataset.eval_df, dataset.pipeline_df)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "measure": "Cross-CPSE duplicate groups in data",
                        "value": summary["cross_cpse_groups"],
                    },
                    {
                        "measure": "Cross-CPSE duplicate pairs recovered",
                        "value": metrics["cross_cpse_true_positives"],
                    },
                    {
                        "measure": "Cross-CPSE recall",
                        "value": round(metrics["cross_cpse_recall"], 3),
                    },
                    {
                        "measure": "Cross-CPSE precision",
                        "value": round(metrics["cross_cpse_precision"], 3),
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No GroundTruth_Group column in this dataset: accuracy cannot be "
            "validated. The counts above (materials scanned, records "
            "clustered, duplicate rate, cross-CPSE clusters) are still real "
            "outputs of this matching run."
        )

    st.subheader("Duplicate clusters by CPSE")
    counts: dict[str, int] = {}
    for material_cluster in result.clusters:
        for cpse in material_cluster.cpses:
            counts[cpse] = counts.get(cpse, 0) + 1
    chart = pd.DataFrame(
        sorted(counts.items(), key=lambda kv: -kv[1]), columns=["CPSE", "clusters"]
    ).set_index("CPSE")
    st.bar_chart(chart)

    st.subheader("Estimated procurement impact")
    eliminated = max(0, result.n_clustered_records - len(result.clusters))
    carrying = (
        eliminated
        * config.AVG_LINE_ITEM_VALUE_INR
        * config.DUPLICATE_CARRYING_COST_PCT
    )
    leverage = (
        eliminated
        * config.AVG_LINE_ITEM_VALUE_INR
        * config.CONSOLIDATION_PRICE_BENEFIT_PCT
    )

    columns = st.columns(3)
    columns[0].metric("Redundant codes removable", f"{eliminated:,}")
    columns[1].metric("Inventory carrying saving", f"INR {carrying/1e7:.2f} cr")
    columns[2].metric("Consolidation leverage", f"INR {leverage/1e7:.2f} cr")

    st.warning(
        "**This is an estimate, not a finding.** It multiplies the redundant "
        f"codes found by three assumed constants: average annual spend per code "
        f"(INR {config.AVG_LINE_ITEM_VALUE_INR:,}), duplicate carrying cost "
        f"({config.DUPLICATE_CARRYING_COST_PCT:.0%}) and consolidation price "
        f"benefit ({config.CONSOLIDATION_PRICE_BENEFIT_PCT:.0%}). All three are "
        "illustrative planning figures set in `src/config.py`, not audited "
        "procurement data. The CPSE assignment in this dataset is synthetic "
        "scaffolding, so these figures demonstrate the calculation, not a real "
        "saving."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the sidebar and dispatch to the selected view."""
    session_state_defaults()
    _apply_pending_view_change()

    st.sidebar.title("National Unified Material Master")
    st.sidebar.caption("AI-driven standardization of material codes across CPSEs")
    view = st.sidebar.radio("View", VIEWS, key="view_choice")

    st.sidebar.divider()
    if st.sidebar.button("Start over / load a different dataset"):
        _reset_session()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Semantic backend: `{config.SEMANTIC_BACKEND}`\n\n"
        f"Auto-approve precision target: {config.AUTO_APPROVE_PRECISION_TARGET:.0%}"
    )
    if st.session_state.decisions:
        st.sidebar.metric("Decisions this session", len(st.session_state.decisions))

    {
        VIEWS[0]: view_load_data,
        VIEWS[1]: view_problem,
        VIEWS[2]: view_run_matching,
        VIEWS[3]: view_review_match,
        VIEWS[4]: view_human_review,
        VIEWS[5]: view_national_code,
        VIEWS[6]: view_dashboard,
    }[view]()


if __name__ == "__main__":
    main()
