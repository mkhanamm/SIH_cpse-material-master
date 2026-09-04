"""
Deterministic match explanation generator (Spec section 4.7).

WHAT THIS FILE DOES
    Renders a human-readable justification for every proposed match, from
    numbers already computed upstream -- no live LLM call.

    The output *structure* is modelled on LLM-based entity-matching explanation
    work (per-attribute alignment + per-channel scores + overall verdict), but
    the implementation is a fixed template. This is a deliberate engineering
    choice for demo reliability: no API key, no network dependency, no cost, no
    non-determinism, and identical output every time a judge clicks the button.
    Swapping in a live LLM for richer prose is a one-function change
    (`render_explanation` is the only seam) -- documented as an upgrade path,
    not hidden as a limitation.

    Example output:
        Match: HIGH confidence (fused score 0.91)
        + Material grade identical (ASTM A106)
        + Dimension match within tolerance (40 mm NB vs 40 mm NB)
        + Description similarity: semantic 0.88 / string 0.94
        - Specification differs (A106 vs A333)
        ? Operating parameter unknown for both records

INPUTS
    A scored pair or cluster + its SimilarityScores and AttributeSets.

OUTPUTS
    Explanation - .verdict, .lines, .as_text(), .as_dict()  (dict form is what
                  gets written to the audit trail, so explanations are
                  reproducible evidence, not UI-only decoration).

KEY FUNCTIONS
    explain_pair(pair, scores, attrs_a, attrs_b) -> Explanation
    explain_cluster(cluster)                     -> Explanation
    render_explanation(explanation)              -> str   (the LLM swap seam)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .similarity import AGREE, CONFLICT, UNKNOWN, _is_missing

# Marker glyphs. Kept ASCII so the same string renders identically in a
# terminal, a Streamlit table and a CSV audit export.
SUPPORT, AGAINST, MISSING, INFO = "+", "-", "?", "="

# Human labels for the machine attribute names. A reviewer reads
# "Nominal bore", not "nominal_size_mm".
ATTRIBUTE_LABELS: dict[str, str] = {
    "nominal_size_mm": "Nominal bore",
    "thickness_mm": "Thickness",
    "width_mm": "Width",
    "rating_kva": "Rating",
    "voltage_kv": "Voltage",
    "power_hp": "Power",
    "flow_m3hr": "Flow rate",
    "head_m": "Head",
    "capacity_cum": "Capacity",
    "area_sqmm": "Conductor area",
    "ply": "Ply count",
    "grade": "Material grade",
    "spec_standard": "Specification",
    "schedule": "Schedule",
    "pressure_class": "Pressure class",
    "ratio": "Gear ratio",
    "material_of_construction": "Material of construction",
}

ATTRIBUTE_UNITS: dict[str, str] = {
    "nominal_size_mm": "mm",
    "thickness_mm": "mm",
    "width_mm": "mm",
    "rating_kva": "kVA",
    "voltage_kv": "kV",
    "power_hp": "HP",
    "flow_m3hr": "m3/hr",
    "head_m": "m",
    "capacity_cum": "cu.m",
    "area_sqmm": "sq.mm",
}


@dataclass
class ExplanationLine:
    """One piece of evidence for or against a match.

    Attributes:
        marker: SUPPORT, AGAINST, MISSING or INFO.
        text: Rendered sentence shown to the reviewer.
        attribute: Attribute this line concerns, when applicable.
        weight: Rough importance, used only to order lines so the decisive
            evidence appears first. Not a model coefficient.
    """

    marker: str
    text: str
    attribute: str | None = None
    weight: float = 0.0


@dataclass
class Explanation:
    """A complete, reproducible justification for one proposed match.

    Attributes:
        verdict: Confidence tier being justified.
        score: The score the verdict rests on.
        lines: Ordered evidence.
        subject: What is being explained -- a pair or a cluster id.
        channel_scores: Per-channel similarity, shown so a reviewer can see
            whether the match rests on text or on structured attributes.
        caveats: Statements limiting how far the match should be trusted.
    """

    verdict: str
    score: float
    lines: list[ExplanationLine] = field(default_factory=list)
    subject: str = ""
    channel_scores: dict[str, float] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        """Render the explanation as plain text.

        Returns:
            A multi-line string suitable for a terminal, a CSV cell or a log.
        """
        return render_explanation(self)

    def as_dict(self) -> dict[str, object]:
        """Serialise for the audit trail.

        The dict form is what governance stores, so a decision made in 2026 can
        be re-read in 2031 with its reasoning intact -- explanations are
        evidence, not UI decoration.

        Returns:
            A JSON-serialisable dict.
        """
        return {
            "verdict": self.verdict,
            "score": round(self.score, 4),
            "subject": self.subject,
            "channel_scores": {k: round(v, 4) for k, v in self.channel_scores.items()},
            "lines": [asdict(line) for line in self.lines],
            "caveats": list(self.caveats),
        }

    def supporting(self) -> list[ExplanationLine]:
        """Evidence in favour of the match.

        Returns:
            Lines marked SUPPORT.
        """
        return [line for line in self.lines if line.marker == SUPPORT]

    def opposing(self) -> list[ExplanationLine]:
        """Evidence against the match.

        Returns:
            Lines marked AGAINST.
        """
        return [line for line in self.lines if line.marker == AGAINST]


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------
def format_value(attribute: str, value: object) -> str:
    """Render an attribute value the way a procurement officer would write it.

    Args:
        attribute: Attribute name.
        value: The extracted value.

    Returns:
        A formatted string, or ``"not stated"`` when the value is absent.
    """
    if _is_missing(value):
        return "not stated"
    if isinstance(value, float):
        rendered = f"{value:g}"
    else:
        rendered = str(value)
    unit = ATTRIBUTE_UNITS.get(attribute)
    return f"{rendered} {unit}" if unit else rendered.upper()


def label_for(attribute: str) -> str:
    """Human-readable label for a machine attribute name.

    Args:
        attribute: Attribute name.

    Returns:
        A display label.
    """
    return ATTRIBUTE_LABELS.get(attribute, attribute.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Pair explanations
# ---------------------------------------------------------------------------
def explain_pair(
    pair_row: pd.Series,
    attrs_a: dict[str, object],
    attrs_b: dict[str, object],
    verdict: str,
    descriptions: tuple[str, str] | None = None,
    hard_conflict_attributes: tuple[str, ...] = (),
) -> Explanation:
    """Explain why two records were or were not judged equivalent.

    Every line is derived from a number computed upstream; nothing here
    recomputes similarity, so the explanation cannot drift from the decision it
    describes.

    Args:
        pair_row: A row of the scored-pairs frame, carrying the channel scores
            and the per-attribute ``flags`` dict.
        attrs_a: Extracted attributes of the first record.
        attrs_b: Extracted attributes of the second record.
        verdict: Confidence tier being justified.
        descriptions: Optional ``(raw_a, raw_b)`` for context.
        hard_conflict_attributes: Attributes whose disagreement is
            disqualifying, so the line can say so explicitly.

    Returns:
        A populated :class:`Explanation`.
    """
    score = float(pair_row.get("match_probability", pair_row.get("fused", 0.0)))
    channel_scores = {
        "semantic": float(pair_row.get("semantic", 0.0)),
        "string": float(pair_row.get("string", 0.0)),
        "attribute": float(pair_row.get("attribute", 0.0)),
    }

    explanation = Explanation(
        verdict=verdict,
        score=score,
        subject=f"records {pair_row['idx_a']} and {pair_row['idx_b']}",
        channel_scores=channel_scores,
    )

    if descriptions:
        explanation.lines.append(
            ExplanationLine(
                INFO, f'A: "{descriptions[0]}"', weight=-1.0
            )
        )
        explanation.lines.append(
            ExplanationLine(INFO, f'B: "{descriptions[1]}"', weight=-1.0)
        )

    flags = pair_row.get("flags") or {}
    for attribute, flag in flags.items():
        label = label_for(attribute)
        left = format_value(attribute, attrs_a.get(attribute))
        right = format_value(attribute, attrs_b.get(attribute))

        if flag == AGREE:
            same = left == right
            explanation.lines.append(
                ExplanationLine(
                    SUPPORT,
                    f"{label} matches ({left})"
                    if same
                    else f"{label} matches within tolerance ({left} vs {right})",
                    attribute,
                    weight=3.0,
                )
            )
        elif flag == CONFLICT:
            disqualifying = attribute in hard_conflict_attributes
            explanation.lines.append(
                ExplanationLine(
                    AGAINST,
                    f"{label} differs ({left} vs {right})"
                    + (" - procurement-critical" if disqualifying else ""),
                    attribute,
                    weight=5.0 if disqualifying else 2.0,
                )
            )
        elif flag == UNKNOWN:
            # Only worth a line when exactly ONE side states the attribute --
            # that is a real asymmetry a reviewer can act on (go and check the
            # other record). When neither states it there is nothing to say, and
            # emitting a line per absent attribute buries the actual evidence
            # under a dozen "not stated" rows.
            # NOTE: extracted values arrive as NaN, not None, so an `is not None`
            # test passes for absent values and produced exactly that flood.
            has_a = not _is_missing(attrs_a.get(attribute))
            has_b = not _is_missing(attrs_b.get(attribute))
            if has_a != has_b:
                stated_by, value = ("A", left) if has_a else ("B", right)
                explanation.lines.append(
                    ExplanationLine(
                        MISSING,
                        f"{label} stated only by record {stated_by} ({value}) "
                        "- not comparable",
                        attribute,
                        weight=1.0,
                    )
                )

    explanation.lines.append(
        ExplanationLine(
            INFO,
            f"Description similarity: semantic {channel_scores['semantic']:.2f} / "
            f"string {channel_scores['string']:.2f}",
            weight=0.5,
        )
    )

    comparable = int(pair_row.get("n_comparable_attrs", 0))
    if comparable == 0:
        explanation.caveats.append(
            "No structured attribute could be compared -- both records state "
            "too little for the match to rest on anything but description text."
        )
    explanation.lines.sort(key=lambda line: -line.weight)
    return explanation


# ---------------------------------------------------------------------------
# Cluster explanations
# ---------------------------------------------------------------------------
def explain_cluster(
    material_cluster,
    df: pd.DataFrame,
    scored_pairs: pd.DataFrame | None = None,
    attribute_lookup: dict[int, dict[str, object]] | None = None,
    hard_conflict_attributes: tuple[str, ...] = (),
) -> Explanation:
    """Explain why a group of records was proposed as one material.

    A cluster explanation leads with its weakest internal link, because that
    edge is what a reviewer should scrutinise: a cluster is only as sound as its
    least convincing member.

    Args:
        material_cluster: A ``matching_engine.MaterialCluster``.
        df: Pipeline frame, for CPSE codes and descriptions.
        scored_pairs: Scored pairs, used to explain the weakest edge in detail.
        attribute_lookup: Extracted attributes keyed by record index.
        hard_conflict_attributes: Procurement-critical attribute names.

    Returns:
        A populated :class:`Explanation`.
    """
    explanation = Explanation(
        verdict=material_cluster.tier,
        score=material_cluster.min_score,
        subject=f"cluster {material_cluster.cluster_id}",
        channel_scores={
            "cluster_mean": material_cluster.mean_score,
            "cluster_weakest_link": material_cluster.min_score,
        },
    )

    explanation.lines.append(
        ExplanationLine(
            INFO,
            f"{material_cluster.size} records proposed as one material, "
            f"from {len(material_cluster.cpses)} CPSE(s): "
            f"{', '.join(material_cluster.cpses)}",
            weight=10.0,
        )
    )

    if material_cluster.is_cross_cpse:
        explanation.lines.append(
            ExplanationLine(
                SUPPORT,
                "Cross-CPSE duplication: the same material is separately coded "
                "by more than one enterprise, which is the case this system "
                "exists to find",
                weight=9.0,
            )
        )

    for index in material_cluster.members:
        explanation.lines.append(
            ExplanationLine(
                INFO,
                f"{df.at[index, 'CPSE']} / {df.at[index, 'CPSE Material Code']}: "
                f'"{df.at[index, "Raw Description"]}"',
                weight=8.0,
            )
        )

    explanation.lines.append(
        ExplanationLine(
            INFO,
            f"Internal agreement: weakest link {material_cluster.min_score:.3f}, "
            f"mean {material_cluster.mean_score:.3f} "
            f"across {len(material_cluster.edges)} comparison(s)",
            weight=7.0,
        )
    )

    if material_cluster.tier_reason:
        explanation.lines.append(
            ExplanationLine(INFO, material_cluster.tier_reason, weight=6.5)
        )

    # Detail the weakest edge -- the one a reviewer should look at hardest.
    if scored_pairs is not None and attribute_lookup and material_cluster.edges:
        weakest = min(material_cluster.edges, key=lambda edge: edge[2])
        a, b, _ = weakest
        match = scored_pairs[
            (scored_pairs["idx_a"] == min(a, b))
            & (scored_pairs["idx_b"] == max(a, b))
        ]
        if len(match):
            detail = explain_pair(
                match.iloc[0],
                attribute_lookup.get(a, {}),
                attribute_lookup.get(b, {}),
                verdict=material_cluster.tier,
                hard_conflict_attributes=hard_conflict_attributes,
            )
            explanation.lines.append(
                ExplanationLine(
                    INFO, "Weakest link, attribute by attribute:", weight=6.0
                )
            )
            for line in detail.lines:
                if line.marker in (SUPPORT, AGAINST, MISSING):
                    explanation.lines.append(
                        ExplanationLine(
                            line.marker, line.text, line.attribute, line.weight - 0.1
                        )
                    )
            explanation.caveats.extend(detail.caveats)

    if material_cluster.tier == "UNKNOWN":
        explanation.caveats.append(
            "Routed to manual review for insufficient data, not because the "
            "descriptions disagree. Confirm against a drawing or spec sheet."
        )

    explanation.lines.sort(key=lambda line: -line.weight)
    return explanation


# ---------------------------------------------------------------------------
# Rendering -- the single seam where a live LLM would be swapped in
# ---------------------------------------------------------------------------
def render_explanation(explanation: Explanation) -> str:
    """Render an :class:`Explanation` as text.

    THIS IS THE LLM SWAP SEAM. Everything upstream produces structured evidence;
    this function turns it into prose. A production upgrade replaces the body
    with a call passing ``explanation.as_dict()`` to a language model for a
    fluent natural-language rationale. Nothing else in the system changes,
    because no other module renders explanations.

    The template is used for the hackathon build deliberately: it needs no API
    key, no network, costs nothing, and produces byte-identical output on every
    run -- which matters when the thing is being demonstrated live.

    Args:
        explanation: The structured explanation.

    Returns:
        Multi-line plain text.
    """
    header = (
        f"Match: {explanation.verdict} confidence "
        f"(score {explanation.score:.2f}) - {explanation.subject}"
    )
    body = [f"  {line.marker} {line.text}" for line in explanation.lines]
    caveats = [f"  ! {caveat}" for caveat in explanation.caveats]
    return "\n".join([header, *body, *caveats])


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from . import (
        attribute_extraction,
        blocking,
        classifier,
        config,
        ingestion,
        matching_engine,
        similarity,
    )
    from .normalization import normalize_series

    dataset = ingestion.load_dataset()
    extracted = attribute_extraction.extract_frame(dataset.pipeline_df)
    joined = dataset.pipeline_df.join(extracted)
    text = normalize_series(joined[config.INPUT_TEXT_COLUMN])

    candidates = blocking.candidate_pairs(joined)
    scored = similarity.score_pairs(joined, candidates.pairs, text)
    model, report = classifier.train(scored, dataset.eval_df)
    scored["match_probability"] = classifier.predict_proba(model, scored)

    tiers = classifier.calibrate_tiers(
        classifier.label_pairs(scored, dataset.eval_df), scored["match_probability"]
    )
    result = matching_engine.run_matching(
        scored, joined, tiers, edge_threshold=0.55,
        attribute_counts=extracted["n_known_attributes"],
    )
    lookup = {
        idx: row.to_dict()
        for idx, row in extracted[
            attribute_extraction.ALL_ATTRIBUTES
        ].iterrows()
    }

    for item in [c for c in result.clusters if c.is_cross_cpse][:2]:
        print(
            explain_cluster(
                item, joined, scored, lookup,
                hard_conflict_attributes=tuple(config.EXACT_MATCH_ATTRIBUTES),
            ).as_text()
        )
        print()
