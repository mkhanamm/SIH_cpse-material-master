"""Unit tests for src.cnmc_generator -- code schema, mapping integrity, merge/split, round-trip."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cnmc_generator import (
    ACTIVE,
    SUPERSEDED,
    CNMCRegistry,
    category_short_code,
    export_mapping,
    import_mapping,
)


class Stub:
    """Minimal stand-in for a MaterialCluster."""

    def __init__(self, members, tier="HIGH"):
        self.members = members
        self.tier = tier


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sector": ["Oil & Gas"] * 4,
            "CPSE": ["ONGC", "BPCL", "IOCL", "HPCL"],
            "CPSE Material Code": [
                "ONGC-PIP-00001", "BPCL-PIP-00002",
                "IOCL-PIP-00003", "HPCL-PIP-00004",
            ],
            "Legacy_Sector_Code": ["OG-1", "OG-2", "OG-3", "OG-4"],
            "Material Category": ["Seamless Pipe"] * 4,
            "Raw Description": [
                "PIPE CS 40NB SCH 20",
                "SEAMLESS CARBON STEEL PIPE 40 MM DIA SCH-20 ASTM A106",
                "CS SMLS PIPE 40NB",
                "M.S. SMLS PIPE DN40 SCH20",
            ],
        }
    )


class TestCodeSchema:
    def test_code_format(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        assert entry.cnmc.startswith("NM-OG-")
        assert entry.cnmc.split("-")[-1].isdigit()
        assert len(entry.cnmc.split("-")) == 4

    def test_sequence_is_not_recycled(self, frame):
        registry = CNMCRegistry()
        first = registry.assign(Stub([0, 1]), frame)
        second = registry.assign(Stub([2, 3]), frame)
        assert first.cnmc != second.cnmc

    def test_multi_word_category_uses_distinct_letters(self):
        """'Seamless Pipe' must not collide with every other 'Sea...' category."""
        assert category_short_code("Seamless Pipe") == "SEP"
        assert category_short_code("Ball Valve") == "BAV"

    def test_single_word_category(self):
        assert category_short_code("Bearing") == "BEA"

    def test_empty_category_is_safe(self):
        assert category_short_code("") == "GEN"

    def test_short_codes_are_three_characters(self):
        for category in ("Pump", "Bearing", "Seamless Pipe", "Heavy Duty Gear Box"):
            assert len(category_short_code(category)) == 3


class TestAssignment:
    def test_members_are_recorded_with_identity(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        assert {m.cpse for m in entry.members} == {"ONGC", "BPCL"}
        assert entry.is_cross_cpse

    def test_cpse_codes_are_never_altered(self, frame):
        """The mapping is additive; enterprises keep transacting on their codes."""
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        assert [m.cpse_material_code for m in entry.members] == [
            "ONGC-PIP-00001",
            "BPCL-PIP-00002",
        ]

    def test_double_assignment_rejected(self, frame):
        registry = CNMCRegistry()
        registry.assign(Stub([0, 1]), frame)
        with pytest.raises(ValueError, match="already mapped"):
            registry.assign(Stub([1, 2]), frame)

    def test_canonical_description_is_a_real_one(self, frame):
        """Never a generated string -- it must appear in some CPSE's records."""
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        assert entry.canonical_description in list(frame["Raw Description"])


class TestLookup:
    def test_forward_lookup(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        assert len(registry.lookup(entry.cnmc)) == 2

    def test_reverse_lookup(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        assert registry.reverse_lookup("ONGC", "ONGC-PIP-00001") == entry.cnmc

    def test_unknown_code_returns_none(self, frame):
        assert CNMCRegistry().reverse_lookup("ONGC", "NOPE") is None

    def test_lookup_of_unknown_cnmc_is_empty(self):
        assert CNMCRegistry().lookup("NM-OG-SEP-99999") == []


class TestMerge:
    def test_absorbed_code_is_superseded_not_deleted(self, frame):
        """A code cited on a purchase order must stay resolvable forever."""
        registry = CNMCRegistry()
        keep = registry.assign(Stub([0, 1]), frame)
        absorb = registry.assign(Stub([2, 3]), frame)
        registry.merge(keep.cnmc, absorb.cnmc)
        assert registry.entries[absorb.cnmc].status == SUPERSEDED
        assert absorb.cnmc in registry.entries

    def test_reverse_lookup_follows_supersession(self, frame):
        registry = CNMCRegistry()
        keep = registry.assign(Stub([0, 1]), frame)
        absorb = registry.assign(Stub([2, 3]), frame)
        registry.merge(keep.cnmc, absorb.cnmc)
        assert registry.reverse_lookup("IOCL", "IOCL-PIP-00003") == keep.cnmc

    def test_survivor_absorbs_members(self, frame):
        registry = CNMCRegistry()
        keep = registry.assign(Stub([0, 1]), frame)
        absorb = registry.assign(Stub([2, 3]), frame)
        assert len(registry.merge(keep.cnmc, absorb.cnmc).members) == 4

    def test_self_merge_rejected(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        with pytest.raises(ValueError, match="into itself"):
            registry.merge(entry.cnmc, entry.cnmc)

    def test_unknown_code_rejected(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        with pytest.raises(KeyError):
            registry.merge(entry.cnmc, "NM-OG-SEP-99999")


class TestSplit:
    def test_split_supersedes_original_and_mints_new(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1, 2, 3]), frame)
        created = registry.split(entry.cnmc, [[0, 1], [2, 3]], frame)
        assert len(created) == 2
        assert registry.entries[entry.cnmc].status == SUPERSEDED
        assert all(c.cnmc != entry.cnmc for c in created)

    def test_partition_must_be_exact(self, frame):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1, 2, 3]), frame)
        with pytest.raises(ValueError, match="partition"):
            registry.split(entry.cnmc, [[0, 1]], frame)

    def test_unknown_code_rejected(self, frame):
        with pytest.raises(KeyError):
            CNMCRegistry().split("NM-OG-SEP-99999", [[0]], frame)


class TestStatsAndRoundTrip:
    def test_stats_count_eliminated_codes(self, frame):
        registry = CNMCRegistry()
        registry.assign(Stub([0, 1, 2]), frame)
        stats = registry.stats()
        assert stats["records_mapped"] == 3
        assert stats["active_codes"] == 1
        assert stats["codes_eliminated"] == 2

    def test_csv_round_trip_preserves_mapping(self, frame, tmp_path):
        registry = CNMCRegistry()
        entry = registry.assign(Stub([0, 1]), frame)
        path = export_mapping(registry, tmp_path / "map.csv")
        restored = import_mapping(path, frame)
        assert set(restored.entries) == set(registry.entries)
        assert restored.reverse_lookup("ONGC", "ONGC-PIP-00001") == entry.cnmc

    def test_round_trip_does_not_recycle_sequence(self, frame, tmp_path):
        registry = CNMCRegistry()
        registry.assign(Stub([0, 1]), frame)
        restored = import_mapping(
            export_mapping(registry, tmp_path / "map.csv"), frame
        )
        assert restored.assign(Stub([2, 3]), frame).cnmc not in registry.entries

    def test_missing_file_raises(self, frame, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_mapping(tmp_path / "absent.csv", frame)

    def test_frame_has_one_row_per_member(self, frame):
        registry = CNMCRegistry()
        registry.assign(Stub([0, 1, 2]), frame)
        assert len(registry.to_frame()) == 3
