# Architecture and Design Rationale

Companion to [`../README.md`](../README.md). The README states what the system does and how well it works; this document explains why it is built the way it is, which decisions were forced by measurement, and what would change at national scale.

---

## 1. How each module maps to the research

The matching sub-problem is solved in the literature. The design applies four established findings rather than re-deriving them.

### 1.1 Two channels, because neither is sufficient

A 2026 comparison of SBERT against classic string similarity on real industrial material records found the two catch **different** duplicate types: embeddings win on paraphrase and word reordering, string methods win on abbreviation and unit variants. Neither dominates.

`similarity.py` therefore runs both and fuses them. The initial weighting (`string` 0.40, `semantic` 0.35, `attribute` 0.25) leans toward the string channel because inspection of `Raw Description` in *this* dataset shows the duplicate signal is dominated by abbreviation variance — `BFLY V/V` ↔ `BUTTERFLY VALVE`, `BRG` ↔ `BEARING`, `SMLS` ↔ `SEAMLESS` — rather than genuine paraphrase. The weights are configurable and the trained classifier learns its own combination regardless; the hand weights matter only for the no-model path.

### 1.2 Explanations structured as the LLM work structures them

Peeters, Steiner & Bizer (EDBT 2025) showed GPT-4-class models matching fine-tuned matchers with zero task-specific training data, and — the part this project borrows — generating attribute-level explanations that correlate with actual similarity metrics.

`explanation.py` takes the **structure** (per-attribute alignment, per-channel scores, overall verdict) and implements it deterministically. This is a demo-reliability decision, not a capability limit: no API key, no network dependency, no per-call cost, and byte-identical output every run, which matters when a judge is watching. `render_explanation` is the only function in the codebase that turns structured evidence into prose, so substituting a live model is a single-function change.

### 1.3 Blocking is mandatory

Standard practice, applied and measured rather than asserted. See §3.

### 1.4 The gap this fills

No published pipeline runs end-to-end for CPSE-style material masters — ingestion through national code generation, human governance, audit trail and dashboard. The assembly is the contribution; the matcher is not.

---

## 2. Structural decisions

### 2.1 Leakage is prevented by structure, not discipline

`Standardized Description` is a cleaned version of the input text and `GroundTruth_Group` is an exact-match map over it (verified: 3,965 groups, 3,965 distinct standardized descriptions, zero groups containing two). Either would hand the matcher its answer.

`ingestion.load_dataset` therefore returns two frames. The evaluation columns are **physically absent** from `pipeline_df`, and `assert_no_leakage()` raises if they reappear. A convention that says "don't use these columns" fails the first time someone writes `df.merge(...)` without thinking; a frame that does not contain them cannot.

### 2.2 The rule layer stays separate from the model layer

`normalization.py` and `attribute_extraction.py` are deterministic and inspectable. A domain expert at a CPSE can read the abbreviation table, disagree with an entry, and change it without touching the pipeline — it is data, not code. Every entry in that table was seeded from a token-frequency scan of the actual descriptions (minimum 8 occurrences), not from general engineering vocabulary, so nothing in it is dead weight.

This matters for adoption more than for accuracy. A reviewer asked to approve a merge that will propagate to every CPSE will not accept "the embedding said so." They will accept "grade, bore and schedule all match, and here is the rule that read them."

### 2.3 Missing data is neutral, not negative

Nulls dominate this dataset: `Operating Parameter` absent in 4,650 of 5,008 records, `Specification/Standard` in 4,342. An attribute missing on either side scores **0.5** — neutral — never 0. Absence of evidence is not evidence of difference, and scoring it as a mismatch would punish records for being under-documented.

The counterpart is that sparse records cannot be auto-approved. `MIN_KNOWN_ATTRIBUTES_FOR_AUTO` routes any cluster whose poorest-documented member states fewer than two structured attributes to `UNKNOWN` — "insufficient data, needs manual review" — regardless of how similar the text reads. 508 records yield zero attributes; a high text score between two bare descriptions is evidence that both are bare, not that they match.

### 2.4 Everything tunable lives in `config.py`

Not because it is tidy, but because a reader asking "why 0.55?" should find the answer in one place with its justification beside it. Where a constant was chosen by measurement, the measurement is in the comment — `BLOCKING_KEYS` carries the full scheme comparison table.

---

## 3. Decisions forced by measurement

Four design choices were wrong on the first attempt. Each is documented here because the correction is more informative than the final value.

### 3.1 Nominal bore is a designation, not a measurement

`1.5 INCH NB` standardizes to `40 mm NB`, not 38.1 mm. Nominal bore is a naming convention that stopped tracking physical dimensions decades ago. Converting arithmetically places the record 5% from every 40 mm peer, and no sane tolerance recovers it.

`INCH_TO_NB_MM` is therefore a lookup table. It is the single highest-value piece of domain knowledge in the extraction layer, and precisely the kind of rule a pure-embedding approach cannot represent.

### 3.2 CPSEs disagree on taxonomy, not just wording

Exact-category blocking could not reach 47 true pairs. Inspecting them showed none was an attribute problem — every one was a category-naming disagreement: `Conveyor Component` ↔ `Conveyor Idler` (32 pairs), `Globe Valve` ↔ `Valve` (12), `Gate Valve` ↔ `Valve` (3).

The `category_token` key emits one bucket per meaningful category token, so those records meet under `conveyor` and `valve`. Generic head nouns are stoplisted — `component` spans eight categories and ~900 records, and bucketing on it would pair a boiler part with a conveyor part for no recall gain. Cost: 11% more comparisons. Benefit: the recall ceiling goes from 98.21% to 100%.

The originally-planned `attribute_signature` key survives in the codebase, benchmarked, defaulting **off**: it adds 476 comparisons and zero pairs here. All five sectors draw category names from one vocabulary, which is an artifact of the synthetic data. Real CPSE masters maintain disjoint taxonomies, where a physical-attribute fingerprint is the only key that can bridge them.

### 3.3 The pair-optimal threshold is the wrong clustering threshold

This is the largest single correction in the project. The classifier's F1-optimal cut-off is **0.06**; the cluster-level optimum is **0.55**.

A cluster asserts equivalence between every pair of its members, including pairs never directly scored. One spurious edge joining two correct clusters of five manufactures 25 false pairs. Errors amplify transitively, so the edge threshold must be far stricter than pair-level F1 suggests. Reusing the pair threshold scored precision 0.185.

Two further traps appeared while fixing it:

- **Clustering only the tuning subsample understates chaining.** Density drives chaining risk, and a 15% subsample has far fewer neighbours to chain through. Tuning on the subsample graph selected 0.15, worth precision 0.44 at full scale. The graph is now built at full corpus density — an unlabelled property available at inference time, so no leakage — with only the *metric* restricted.
- **Restricting the metric to validation-only pairs overstates precision**, 0.86 against a true 0.55. False positives are cross-group pairs, and with 15% of groups held out, two validation groups rarely chain into the same cluster; the sample is starved of exactly the errors it exists to measure. Tuning now scores over all non-test records. Test labels are never touched either way.

`MIN_CLUSTER_COHESION_RATIO` guards the same failure from the other side: a component whose mean internal weight falls below the floor is split by removing its weakest edges. At the chosen threshold no component needs splitting, which is the desired outcome — the guard is load-bearing at lower thresholds and idle at this one.

### 3.4 The feedback loop fed the model nothing

`retrain_with_feedback` originally duplicated reviewed rows while leaving their labels derived from `GroundTruth_Group`. The model was therefore re-reading labels it already had. In production there is no ground-truth column at all — the reviewer *is* the label — so the override is the entire mechanism, not a refinement.

Separately, `recalibrate_threshold` fitted a global threshold to an uncertainty-sampled set. That is statistically invalid: the queue deliberately selects pairs near the boundary, so the sample is not representative of the score distribution. Naive fitting dragged the threshold from 0.55 to 0.19 and cost roughly two points of F1. It is now blended (25%) and gated on a minimum label count.

Even corrected, retraining costs 0.006 F1 over 60 decisions, for the same reason: upweighting a boundary-only sample fivefold adds bias without adding information. It is reported as a loss rather than presented as a gain. `apply_decisions` — treating a human verdict as fact rather than as a training hint — is the default path and yields +0.024.

---

## 4. Confidence routing

Four tiers, because "match / no match" cannot express the two distinct reasons for hesitation.

| Tier | Meaning | Action |
|---|---|---|
| `HIGH` | Weakest internal link clears the auto-approve cut-off | CNMC issued automatically |
| `MEDIUM` | Between review and auto-approve cut-offs | Human decides |
| `UNKNOWN` | Too few structured attributes to judge | Human decides, flagged as a *data* problem |
| `LOW` | Below the review cut-off | No match asserted |

`UNKNOWN` is separate from `MEDIUM` deliberately. Both go to a human, but they ask different questions. `MEDIUM` asks "is this judgement right?"; `UNKNOWN` asks "can you find the specification?" Collapsing them would hide that 209 clusters are blocked on missing documentation rather than on model uncertainty — which is a procurement-process finding, not a modelling one.

Promotion to `HIGH` uses the **weakest** internal edge, never the mean. A mean lets one strong pair carry a doubtful third member into an unattended merge.

Cut-offs are **calibrated from a precision target**, not hard-coded. A fixed 0.85/0.65 pair is only meaningful on the scale it was written for, and this system has three in play: the hand-fused score, and classifier probabilities under each semantic backend. `AUTO_APPROVE_PRECISION_TARGET = 0.99` is stricter than the F1-optimal point on purpose — F1 weighs a missed duplicate and a wrong merge equally, and they are not equal. A missed duplicate is found on the next run. A bad merge puts two different materials behind one code across every CPSE that adopts it.

The known weakness in this calibration is recorded in README §6.4.

---

## 5. Governance

The audit log is the source of truth about *what happened*; the registry holds *current state*. Keeping them separate is what allows state to be reconstructed at any past point, and what stops a registry bug from quietly erasing the record of its own mistake.

Three rules follow:

**Rollback appends, never deletes.** Reversing an incorrect auto-approval writes a compensating event cross-linked to the original. Deleting the row would remove the evidence that the error happened, which is exactly what an auditor needs to see.

**Every event carries an actor.** Automated decisions are attributed to a named system actor. "The system did it" is an answer; "nobody did it" is not.

**Every event carries a reason.** For machine decisions this is the serialised `Explanation`, so the log answers "on what evidence?" rather than only "what changed?" — which is why explanations are stored as structured dicts rather than rendered strings.

The same reasoning shapes `cnmc_generator`. Merging supersedes rather than deletes, and reverse lookup follows the supersession chain, because a code cited on a purchase order must stay resolvable forever. Splitting mints new codes rather than reusing the original, so no downstream system sees a code silently change meaning. Sequence numbers are global and never recycled.

---

## 6. What changes at national scale

This prototype handles 5,008 records in 81 seconds. A real national material master is on the order of 10⁷ records across dozens of CPSEs. What breaks, and what does not:

**Blocking holds.** It is the component designed for this. At 10⁷ records the all-pairs count is ~5×10¹³, and a 65× reduction is not enough on its own — the blocking keys would need a third level (locality-sensitive hashing over embeddings, or a vector index) and block sizes would need capping more aggressively than `MAX_BLOCK_SIZE = 400`.

**Pairwise scoring must move off pandas.** `score_pairs` iterates candidate pairs in Python. At 10⁸ candidate pairs this needs vectorisation and a FAISS or ScaNN index for the semantic channel rather than a dense matrix.

**The classifier is fine.** Nine features and gradient-boosted trees scale trivially; it is already the cheapest part of the run.

**Clustering needs partitioning.** `networkx` connected components over 10⁷ nodes is feasible but memory-hungry. Partitioning by blocking key and clustering within partitions is the obvious move, accepting the small recall cost that partition boundaries impose — and measuring it, as §3.2 measures the blocking ceiling.

**Governance becomes the hard part, not the ML.** At national scale the questions are: who authorises a merge that affects seventeen enterprises? What is the appeal route when a CPSE disputes a code? How does a mapping version propagate to seventeen ERPs without a flag day? Those are institutional problems the batch round-trip in §4.12 only gestures at. The technical answer — versioned mappings, append-only audit, compensating rollback — is in place; the process answer is not, and would not be a hackathon deliverable.

**The synthetic CPSE assignment would be replaced by real sourcing history**, which changes the problem: real masters carry plant codes, valuation classes and historic purchase-order text that this dataset does not have, and which would materially improve matching. Real data is also messier in ways this dataset is not — multilingual descriptions, OCR'd spec sheets, decades of inconsistent legacy migration.
