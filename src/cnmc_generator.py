"""
Common National Material Code generation and mapping (Spec section 4.9).

WHAT THIS FILE DOES
    Assigns one stable national code per approved equivalence group and
    maintains the bidirectional mapping back to every contributing CPSE code,
    so nothing is ever lost in the harmonization.

    CNMC schema:  NM-<SS>-<CC>-<NNNNN>
        SS    sector digit-pair   (OG, PW, ST, MN, HE)
        CC    category short code (PIP, BRG, TRF, ...)
        NNNNN zero-padded sequence
    Codes are semantic enough for a procurement officer to read, but the
    sequence -- not the semantics -- is the identity: codes are never recycled
    and never rewritten when attributes change.

INPUTS
    Approved MaterialClusters + the source dataframe.

OUTPUTS
    outputs/cnmc_mapping.csv - the national mapping table
    CNMCRegistry             - in-memory registry with forward/reverse lookup

KEY FUNCTIONS
    CNMCRegistry.assign(cluster)      -> str  (the new CNMC)
    CNMCRegistry.lookup(cnmc)         -> list[MemberRecord]
    CNMCRegistry.reverse_lookup(code) -> str | None
    CNMCRegistry.merge(a, b) / .split(cnmc, groups)
    export_mapping(registry, path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

ACTIVE, SUPERSEDED = "active", "superseded"


@dataclass
class MemberRecord:
    """One CPSE material code mapped to a national code.

    Attributes:
        cpse: Owning enterprise.
        cpse_material_code: The enterprise's own code -- never altered. The
            mapping is additive; a CPSE keeps operating on its existing code
            while the national code provides the cross-enterprise view.
        legacy_sector_code: Prior sector-generic code, kept for traceability.
        record_index: Row index in the source master.
        description: Raw description as supplied.
    """

    cpse: str
    cpse_material_code: str
    legacy_sector_code: str
    record_index: int
    description: str


@dataclass
class CNMCEntry:
    """A national code and everything mapped to it.

    Attributes:
        cnmc: The national code.
        members: Contributing CPSE records.
        status: ``active`` or ``superseded``. Superseded codes are retained,
            never deleted: a code that has appeared in a purchase order must
            remain resolvable forever.
        superseded_by: Successor code when this one was merged away.
        created_at: ISO-8601 UTC creation time.
        created_by: Who or what created it.
        tier: Confidence tier of the originating cluster.
        canonical_description: Representative description for the group.
    """

    cnmc: str
    members: list[MemberRecord] = field(default_factory=list)
    status: str = ACTIVE
    superseded_by: str | None = None
    created_at: str = ""
    created_by: str = "system"
    tier: str = ""
    canonical_description: str = ""

    @property
    def cpses(self) -> list[str]:
        """Distinct CPSEs mapped to this code.

        Returns:
            Sorted CPSE names.
        """
        return sorted({m.cpse for m in self.members})

    @property
    def is_cross_cpse(self) -> bool:
        """Whether this code unifies codes from more than one enterprise.

        Returns:
            True when at least two CPSEs contribute.
        """
        return len(self.cpses) > 1


def category_short_code(category: str) -> str:
    """Derive a three-letter category segment for the national code.

    Args:
        category: Material category name.

    Returns:
        A three-character uppercase code. Multi-word categories take one letter
        per word where possible, so `Seamless Pipe` -> `SEP` rather than a
        collision-prone `SEA`.
    """
    words = [w for w in re.split(r"[^A-Za-z]+", str(category)) if w]
    if not words:
        return "GEN"
    if len(words) == 1:
        return words[0][:3].upper().ljust(3, "X")
    if len(words) == 2:
        return (words[0][:2] + words[1][:1]).upper()
    return "".join(w[0] for w in words[:3]).upper()


class CNMCRegistry:
    """The national mapping table, held in memory and exportable to CSV.

    Every mutation returns enough information for ``governance`` to log it. The
    registry itself deliberately does not write the audit log: that would make
    the two impossible to test independently and would let a caller mutate the
    mapping without an audit entry by importing the wrong module.
    """

    def __init__(self, prefix: str = config.CNMC_PREFIX) -> None:
        """Initialise an empty registry.

        Args:
            prefix: Code prefix, defaults to ``config.CNMC_PREFIX``.
        """
        self.prefix = prefix
        self.entries: dict[str, CNMCEntry] = {}
        self._reverse: dict[tuple[str, str], str] = {}
        self._sequence = 0

    def _next_code(self, sector: str, category: str) -> str:
        """Mint the next unused national code.

        The sequence is global rather than per-category, so a code is unique on
        its own and two codes can never collide if a category is later renamed.

        Args:
            sector: Sector name.
            category: Material category name.

        Returns:
            A new CNMC string.
        """
        self._sequence += 1
        sector_code = config.SECTOR_CODES.get(sector, "XX")
        return (
            f"{self.prefix}-{sector_code}-{category_short_code(category)}-"
            f"{self._sequence:0{config.CNMC_SEQUENCE_WIDTH}d}"
        )

    def assign(
        self,
        material_cluster,
        df: pd.DataFrame,
        created_by: str = "system",
    ) -> CNMCEntry:
        """Assign a national code to an approved cluster.

        Args:
            material_cluster: A ``matching_engine.MaterialCluster``.
            df: Pipeline frame supplying identity columns.
            created_by: Actor to attribute creation to.

        Returns:
            The created :class:`CNMCEntry`.

        Raises:
            ValueError: If any member is already mapped. A record belongs to
                exactly one national code; silently remapping it would break
                the reverse lookup that legacy systems depend on.
        """
        already = [
            (str(df.at[i, "CPSE"]), str(df.at[i, "CPSE Material Code"]))
            for i in material_cluster.members
            if (str(df.at[i, "CPSE"]), str(df.at[i, "CPSE Material Code"]))
            in self._reverse
        ]
        if already:
            raise ValueError(
                f"Records already mapped to a national code: {already[:3]}. "
                "Use merge() to combine existing codes instead of reassigning."
            )

        first = material_cluster.members[0]
        code = self._next_code(
            str(df.at[first, "Sector"]), str(df.at[first, "Material Category"])
        )

        members = [
            MemberRecord(
                cpse=str(df.at[i, "CPSE"]),
                cpse_material_code=str(df.at[i, "CPSE Material Code"]),
                legacy_sector_code=str(df.at[i, "Legacy_Sector_Code"]),
                record_index=int(i),
                description=str(df.at[i, "Raw Description"]),
            )
            for i in material_cluster.members
        ]

        entry = CNMCEntry(
            cnmc=code,
            members=members,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
            tier=getattr(material_cluster, "tier", ""),
            canonical_description=_canonical_description(members),
        )
        self.entries[code] = entry
        for member in members:
            self._reverse[(member.cpse, member.cpse_material_code)] = code
        return entry

    def lookup(self, cnmc: str) -> list[MemberRecord]:
        """Resolve a national code to its member records.

        Args:
            cnmc: The national code.

        Returns:
            Member records, empty when the code is unknown.
        """
        entry = self.entries.get(cnmc)
        return list(entry.members) if entry else []

    def reverse_lookup(self, cpse: str, cpse_material_code: str) -> str | None:
        """Resolve a CPSE's own code to its national code.

        This is the lookup an existing ERP calls during a transition: the
        enterprise keeps using its own code and the national code is resolved
        behind it.

        Args:
            cpse: Owning enterprise.
            cpse_material_code: The enterprise's code.

        Returns:
            The national code, following supersession, or None.
        """
        code = self._reverse.get((cpse, cpse_material_code))
        return self.resolve(code) if code else None

    def resolve(self, cnmc: str) -> str | None:
        """Follow a supersession chain to the currently active code.

        Args:
            cnmc: Any national code, active or superseded.

        Returns:
            The active code, or None if unknown. Guards against a cycle rather
            than looping forever.
        """
        seen: set[str] = set()
        current = cnmc
        while current and current in self.entries:
            entry = self.entries[current]
            if entry.status == ACTIVE:
                return current
            if current in seen:
                return current
            seen.add(current)
            current = entry.superseded_by
        return current if current in self.entries else None

    def merge(self, keep: str, absorb: str, actor: str = "system") -> CNMCEntry:
        """Merge one national code into another.

        The absorbed code is marked superseded and left in place, not deleted.
        Any purchase order, contract or stock record already citing it must stay
        resolvable, so supersession is a redirect rather than an erasure.

        Args:
            keep: Code that survives.
            absorb: Code that is superseded.
            actor: Who performed the merge.

        Returns:
            The surviving entry.

        Raises:
            KeyError: If either code is unknown.
            ValueError: If asked to merge a code into itself.
        """
        if keep == absorb:
            raise ValueError("Cannot merge a national code into itself.")
        for code in (keep, absorb):
            if code not in self.entries:
                raise KeyError(f"Unknown national code {code!r}.")

        survivor, absorbed = self.entries[keep], self.entries[absorb]
        survivor.members.extend(absorbed.members)
        survivor.canonical_description = _canonical_description(survivor.members)
        absorbed.status = SUPERSEDED
        absorbed.superseded_by = keep

        for member in absorbed.members:
            self._reverse[(member.cpse, member.cpse_material_code)] = keep
        return survivor

    def split(
        self, cnmc: str, groups: list[list[int]], df: pd.DataFrame, actor: str = "system"
    ) -> list[CNMCEntry]:
        """Split a national code into several, superseding the original.

        The corrective action for an over-merged cluster. New codes are minted
        rather than reusing the original, so no downstream system silently sees
        a code change meaning underneath it.

        Args:
            cnmc: Code to split.
            groups: Record indices per resulting group.
            df: Pipeline frame.
            actor: Who performed the split.

        Returns:
            The newly created entries.

        Raises:
            KeyError: If the code is unknown.
            ValueError: If the groups do not partition the original membership.
        """
        if cnmc not in self.entries:
            raise KeyError(f"Unknown national code {cnmc!r}.")

        original = self.entries[cnmc]
        original_indices = {m.record_index for m in original.members}
        proposed = [i for group in groups for i in group]
        if sorted(proposed) != sorted(original_indices):
            raise ValueError(
                "Split groups must partition the original membership exactly; "
                f"got {len(proposed)} records for {len(original_indices)}."
            )

        for member in original.members:
            self._reverse.pop((member.cpse, member.cpse_material_code), None)
        original.status = SUPERSEDED

        created: list[CNMCEntry] = []
        for group in groups:
            stub = type("Stub", (), {"members": group, "tier": original.tier})()
            entry = self.assign(stub, df, created_by=actor)
            created.append(entry)
        return created

    def to_frame(self) -> pd.DataFrame:
        """Flatten the registry into the exportable mapping table.

        One row per (national code, CPSE code) pair -- the shape an ERP expects
        for a cross-reference load.

        Returns:
            The mapping table.
        """
        rows = []
        for entry in self.entries.values():
            for member in entry.members:
                rows.append(
                    {
                        "CNMC": entry.cnmc,
                        "Status": entry.status,
                        "Superseded_By": entry.superseded_by or "",
                        "CPSE": member.cpse,
                        "CPSE_Material_Code": member.cpse_material_code,
                        "Legacy_Sector_Code": member.legacy_sector_code,
                        "Raw_Description": member.description,
                        "Canonical_Description": entry.canonical_description,
                        "Confidence_Tier": entry.tier,
                        "Created_At": entry.created_at,
                        "Created_By": entry.created_by,
                    }
                )
        return pd.DataFrame(rows)

    def stats(self) -> dict[str, int]:
        """Summary counts for the dashboard.

        Returns:
            Codes issued, active codes, cross-CPSE codes, records mapped, and
            codes eliminated (records mapped minus active codes -- the actual
            reduction in the national catalogue).
        """
        active = [e for e in self.entries.values() if e.status == ACTIVE]
        records_mapped = sum(len(e.members) for e in active)
        return {
            "codes_issued": len(self.entries),
            "active_codes": len(active),
            "cross_cpse_codes": sum(1 for e in active if e.is_cross_cpse),
            "records_mapped": records_mapped,
            "codes_eliminated": records_mapped - len(active),
        }


def _canonical_description(members: list[MemberRecord]) -> str:
    """Choose a representative description for a group.

    The longest description is used, on the reasoning that the most complete
    wording is the most informative one to show a procurement officer. This is a
    heuristic, not a generated summary: inventing a description would create a
    string that appears in no CPSE's records.

    Args:
        members: Member records.

    Returns:
        The chosen description, or empty string.
    """
    if not members:
        return ""
    return max((m.description for m in members), key=len)


def export_mapping(registry: CNMCRegistry, path: Path | None = None) -> Path:
    """Write the mapping table to CSV.

    Args:
        registry: The registry to export.
        path: Destination; defaults to ``config.MAPPING_TABLE_PATH``.

    Returns:
        The path written.
    """
    path = path or config.MAPPING_TABLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_frame().to_csv(path, index=False)
    return path


def import_mapping(path: Path, df: pd.DataFrame) -> CNMCRegistry:
    """Rebuild a registry from an exported CSV.

    Together with :func:`export_mapping` this is the batch round-trip required
    by Spec section 4.12: a national mapping can leave this system as a flat
    file, be loaded into an ERP, and be read back without loss.

    Args:
        path: CSV produced by :func:`export_mapping`.
        df: Pipeline frame, used to recover record indices.

    Returns:
        The reconstructed registry.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No mapping file at {path}.")

    table = pd.read_csv(path)
    index_by_code = {
        (str(row["CPSE"]), str(row["CPSE Material Code"])): int(idx)
        for idx, row in df[["CPSE", "CPSE Material Code"]].iterrows()
    }

    registry = CNMCRegistry()
    for code, group in table.groupby("CNMC", sort=True):
        first = group.iloc[0]
        entry = CNMCEntry(
            cnmc=str(code),
            status=str(first.get("Status", ACTIVE)),
            superseded_by=str(first.get("Superseded_By") or "") or None,
            created_at=str(first.get("Created_At", "")),
            created_by=str(first.get("Created_By", "import")),
            tier=str(first.get("Confidence_Tier", "")),
            canonical_description=str(first.get("Canonical_Description", "")),
        )
        for _, row in group.iterrows():
            key = (str(row["CPSE"]), str(row["CPSE_Material_Code"]))
            entry.members.append(
                MemberRecord(
                    cpse=key[0],
                    cpse_material_code=key[1],
                    legacy_sector_code=str(row.get("Legacy_Sector_Code", "")),
                    record_index=index_by_code.get(key, -1),
                    description=str(row.get("Raw_Description", "")),
                )
            )
            registry._reverse[key] = str(code)
        registry.entries[str(code)] = entry

    numbers = [
        int(c.rsplit("-", 1)[-1])
        for c in registry.entries
        if c.rsplit("-", 1)[-1].isdigit()
    ]
    registry._sequence = max(numbers, default=0)
    return registry


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    print("Run via app.py or notebooks/evaluation.ipynb.")
