# National Unified Material Master for CPSEs

**Smart India Hackathon 2026 — AI-Driven Standardization and Harmonization of Material Codes Across CPSEs**

---

## 1. The problem

Every Central Public Sector Enterprise maintains its own material master. Nothing obliges ONGC, IOCL and HPCL to describe the same 40 mm seamless pipe the same way, so the same physical item carries a different code and a different free-text description at every enterprise that buys it. One writes `BFLY V/V LUG 300 NB CL-150`; another writes `BUTTERFLY VALVE LUG TYPE 300MM CLASS 150`. They are the same valve. No system knows that.

The cost is not cosmetic. Procurement cannot aggregate demand it cannot see, so volume leverage is lost. Inventory is carried several times over under different codes. Cross-CPSE stock transfer is impossible when neither side can prove the items are equivalent. In the dataset supplied here, **558 duplicate groups span more than one enterprise** — 1,990 pairs of material codes that refer to the same thing and do not know it.

This system ingests material masters from multiple CPSEs, finds those duplicates, and issues one **Common National Material Code (CNMC)** per equivalence group while keeping every original CPSE code traceable and in use.

---

## 2. Architecture

```
                        cpse_synthetic_dataset_with_cpse_mapping.xlsx
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ ingestion.py        Load workbook, SPLIT OFF evaluation labels structurally  │
   │                     pipeline_df (12 cols) │ eval_df (answer key, sealed)     │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ normalization.py    case/punct → abbreviation table → unit canonicalization  │
   │                     "BFLY V/V LUG 300 NB CL-150"                             │
   │                       → "butterfly valve lug 300 nominal bore class 150"     │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ attribute_extraction.py   Deterministic regex → size, grade, spec, schedule, │
   │                           class, capacity …  Nominal-bore lookup, not ×25.4  │
   │                           Missing → "unknown", never fabricated              │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ blocking.py         category_size ∪ category_token  (multi-key)              │
   │                     12,537,528 → 193,924 comparisons · 64.7× · 100% ceiling  │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ similarity.py       SEMANTIC  (SBERT │ offline TF-IDF+SVD)  ─┐               │
   │                     STRING    (Jaro-Winkler + token-set)    ─┼─→ fused score │
   │                     ATTRIBUTE (tolerance / exact match)     ─┘  + conflict   │
   │                                                                   veto       │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ classifier.py       Gradient-boosted trees on the channel scores             │
   │                     GROUP-disjoint train/val/test · threshold tuned on val   │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
   ┌──────────────────────────────────────▼──────────────────────────────────────┐
   │ matching_engine.py  Graph → connected components → cohesion split → tiers    │
   │                     HIGH (auto) · MEDIUM (review) · LOW · UNKNOWN (sparse)   │
   └───────┬──────────────────────────────┬──────────────────────────────┬───────┘
           │                              │                              │
   ┌───────▼────────┐   ┌─────────────────▼──────────────┐   ┌───────────▼───────┐
   │ explanation.py │   │ review_workflow.py             │   │ cnmc_generator.py │
   │ per-attribute  │──▶│ uncertainty-sampled queue      │──▶│ NM-OG-SEP-00001   │
   │ rationale      │   │ approve / reject / edit        │   │ ↔ (CPSE, code)…   │
   │ (deterministic)│   │ → active learning              │   │ merge / split     │
   └───────┬────────┘   └─────────────────┬──────────────┘   └───────────┬───────┘
           │                              │                              │
           └──────────────────────────────▼──────────────────────────────┘
                          ┌───────────────────────────────┐
                          │ governance.py                 │
                          │ append-only audit log         │
                          │ compensating-event rollback   │
                          └───────────────┬───────────────┘
                                          │
                          ┌───────────────▼───────────────┐
                          │ app.py — 6-view Streamlit demo│
                          └───────────────────────────────┘
```

---

## 3. Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+. The dataset ships in `data/`, and a trained model artifact in `models/`, so the app runs immediately with no preparation step.

`models/classifier.pkl` is committed deliberately, not just left over from development: it is what the app loads for a dataset with no `GroundTruth_Group` column (a real CPSE upload), so uploaded datasets get real classifier-based matching out of the box, without a training step. `python -m src.classifier` regenerates it.

To reproduce the pipeline outside the UI:

```bash
python -m src.blocking          # blocking scheme comparison
python -m src.classifier        # train and evaluate, writes models/classifier.pkl
python -m src.matching_engine   # full pipeline + evaluation
python -m pytest tests/ -q      # 229 tests
```

**Semantic backend.** `config.SEMANTIC_BACKEND` selects `"sbert"` (production), `"tfidf_svd"` (offline) or `"auto"`. Every metric below was produced with **`tfidf_svd`**, the offline backend, because the development environment had no access to the model CDN. Both backends are live and interchangeable; the one actually used is recorded on the scored frame and printed with every report, so no figure is ever ambiguous about its source. Switching to SBERT requires no code change — the tier cut-offs are calibrated from a precision target rather than hard-coded, so they re-derive themselves against the new score distribution.

---

## 4. Screenshots

> **Not yet captured.** Run `streamlit run app.py`, walk the six views in order, and save images to `docs/img/`, then replace this section. The views are:
>
> | File | View | What it shows |
> |---|---|---|
> | `docs/img/01-problem.png` | The Problem | Real cross-CPSE duplicates from the loaded data |
> | `docs/img/02-run-matching.png` | Run Matching | Blocking reduction, classifier report, tier routing |
> | `docs/img/03-explanation.png` | Review a Match | Per-attribute rationale for one cluster |
> | `docs/img/04-review.png` | Human Review | Approve/reject/edit + active-learning before/after |
> | `docs/img/05-cnmc.png` | National Code | Mapping table, legacy lookup, audit trail, rollback |
> | `docs/img/06-dashboard.png` | Dashboard | Duplicate rate, cross-CPSE validation, savings estimate |

---

## 5. Evaluation

All figures on the full 5,008-record dataset, semantic backend `tfidf_svd`. Full pipeline runtime: **81 seconds**.

### 5.1 Blocking

| Scheme | Comparisons | Reduction | Recall ceiling | True pairs lost |
|---|---:|---:|---:|---:|
| None (all pairs) | 12,537,528 | 1.0× | 100.00% | 0 |
| `category_size` only | 174,466 | 71.9× | 98.21% | 47 |
| `category_token` only | 185,088 | 67.7× | 97.72% | 60 |
| `attribute_signature` only | 11,123 | 1127.2× | 4.64% | 2,507 |
| **`category_size` + `category_token`** | **193,924** | **64.7×** | **100.00%** | **0** |
| Union of all three | 194,400 | 64.5× | 100.00% | 0 |

A speedup quoted without its recall cost is not a measurement — any scheme can be arbitrarily fast by discarding candidates. The chosen scheme loses nothing.

### 5.2 Trained classifier (pair level, held out on unseen clusters)

Gradient-boosted trees over nine similarity features. Ground-truth **groups**, not pairs, are split three ways; pairs straddling a boundary are discarded rather than assigned. Train 59,404 · validation 4,317 · test 18,272 · discarded 111,931.

| Scope | Precision | Recall | F1 |
|---|---:|---:|---:|
| Overall | 0.711 | 0.894 | 0.792 |
| Cross-CPSE (n=396) | 0.738 | 0.891 | 0.808 |
| Within-CPSE (n=121) | 0.634 | 0.901 | 0.744 |

Average precision **0.917**. Logistic regression, retained as the interpretable baseline, reaches AP 0.508: the decision is a conjunction ("high string similarity **and** zero attribute conflicts") that a single hyperplane cannot express.

### 5.3 Clustering (the number that matters)

604 clusters covering 1,565 records. Tier routing: HIGH 246 · MEDIUM 149 · UNKNOWN 209 · LOW 0.

| Scope | Precision | Recall | F1 | True duplicates |
|---|---:|---:|---:|---:|
| Overall | 0.889 | 0.906 | **0.897** | 2,629 |
| **Cross-CPSE** | **0.898** | **0.912** | **0.905** | 1,990 |
| Within-CPSE | 0.862 | 0.886 | 0.874 | 639 |

Cross-CPSE performance exceeds within-CPSE, which is the useful direction: cross-enterprise duplication is the harder case and the one the problem statement is about. **1,815 of 1,990 cross-CPSE duplicate pairs recovered.**

Restricting to the auto-approved HIGH tier alone: precision 0.917, recall 0.151 — deliberately conservative, since these merge without a human.

### 5.4 Active learning

60 simulated reviewer decisions producing 209 labelled pairs:

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Before review | 0.889 | 0.906 | 0.897 |
| After decisions applied | 0.939 | 0.904 | **0.921** (+0.024) |

Retraining the classifier on the same decisions moves F1 to 0.891 — **slightly worse**, and reported as such. See §6.

### 5.5 National codes issued

246 CNMCs, **193 of them spanning more than one CPSE**, covering 567 material records and eliminating 321 redundant codes from the national catalogue.

---

## 6. Known limitations

Stated plainly, because a judge will find them anyway.

1. **The CPSE assignment in this dataset is synthetic.** Sector and category structure are realistic; which enterprise owns which code was assigned for testing. Nothing here is real procurement history, so the savings figure demonstrates the calculation, not a saving.

2. **All metrics use the offline `tfidf_svd` encoder**, not SBERT. The development environment had no model-CDN access. Both backends are implemented and tested; SBERT is untested end-to-end and its numbers will differ.

3. **Retraining on reviewer feedback does not yet help** (−0.006 F1 over 60 decisions). The same uncertainty sampling that makes the review queue efficient means every reviewed pair sits at the decision boundary, and upweighting a boundary-only sample adds bias without new information. It should pay off in the thousands of decisions; at 60 it does not. `apply_decisions` — applying human corrections directly — is the default feedback path and gives a clean +0.024.

4. **HIGH-tier calibration is approximate.** Cut-offs are derived from a sweep in which each row clusters *at* that threshold, but at runtime clusters form at 0.55 and are tiered by minimum internal score. These are not the same object, and it shows: predicted HIGH precision 0.992, achieved 0.917. Documented rather than papered over; the fix is to sweep tier cut-offs at a fixed edge threshold.

5. **Explanations are templated, not LLM-generated.** A deliberate choice for demo reliability — no API key, no network, no cost, byte-identical output every run. `explanation.render_explanation` is the single seam where a live model substitutes in.

6. **`GroundTruth_Group` is strict.** It is an exact-match map over `Standardized Description`, so an under-specified record ("40 mm NB, Schedule 20") counts as a *non*-duplicate of its fully-specified sibling ("…, ASTM A106"). Some scored false positives are arguably correct merges.

7. **No live ERP connection.** Integration is a batch CSV round-trip (`export_mapping` / `import_mapping`), verified lossless. A claimed live SAP connector would have been fiction.

8. **`attribute_signature` blocking contributes nothing here** and defaults off. All five sectors draw category names from one shared vocabulary — an artifact of the synthetic data. It is retained for real multi-CPSE loads with genuinely disjoint taxonomies.

---

## 7. What is novel here

The matching techniques are not new, and this project does not pretend otherwise. Sentence embeddings, string similarity, blocking and graph clustering are all established, and the design applies published findings directly — running semantic and string channels in parallel because each catches duplicate types the other misses, and structuring explanations the way zero-shot LLM entity-matching work does.

What does not exist in the literature is an end-to-end **governed** pipeline for CPSE-style material masters: ingestion through duplicate detection, national code generation, human review with a measured feedback loop, audit trail with rollback, and a dashboard — as one system a procurement officer could actually operate. That assembly is the contribution.

Three specific pieces earned their place by measurement rather than assumption. **Nominal bore is a designation, not a measurement** — 1.5″ NB is 40 mm, not 38.1 mm, and multiplying by 25.4 loses the match under any sane tolerance. **CPSEs disagree on category taxonomy as much as on descriptions**, which is why blocking needs a token key, not just an exact-category key. And **the pair-optimal threshold is the wrong clustering threshold by an order of magnitude** (0.06 vs 0.55), because a cluster asserts equivalence between every pair of its members and one bad edge joining two correct clusters of five manufactures 25 false pairs.

See [`docs/architecture.md`](docs/architecture.md) for design rationale and [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb) for the reproducible evaluation.
