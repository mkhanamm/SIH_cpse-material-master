"""Unit tests for src.governance -- append-only semantics, compensating rollback, integrity."""

from __future__ import annotations

import pytest

from src.governance import CREATE, MERGE, ROLLBACK, AuditLog


@pytest.fixture
def log(tmp_path) -> AuditLog:
    """A fresh audit log backed by a temporary file."""
    return AuditLog(path=tmp_path / "audit.jsonl", load=False)


class TestRecording:
    def test_event_ids_are_sequential(self, log):
        for _ in range(3):
            log.record(CREATE, "system", "NM-OG-SEP-00001")
        assert [e.event_id for e in log.events] == [0, 1, 2]

    def test_unknown_action_rejected(self, log):
        with pytest.raises(ValueError, match="Unknown action"):
            log.record("obliterate", "system", "NM-OG-SEP-00001")

    def test_missing_actor_rejected(self, log):
        """An unattributed change cannot be audited."""
        with pytest.raises(ValueError, match="needs an actor"):
            log.record(CREATE, "", "NM-OG-SEP-00001")

    def test_whitespace_actor_rejected(self, log):
        with pytest.raises(ValueError, match="needs an actor"):
            log.record(CREATE, "   ", "NM-OG-SEP-00001")

    def test_event_persists_to_disk(self, log):
        log.record(CREATE, "system", "NM-OG-SEP-00001")
        assert log.path.exists()
        assert len(AuditLog(path=log.path).events) == 1

    def test_reason_is_retained(self, log):
        reason = {"verdict": "HIGH", "score": 0.97}
        event = log.record(CREATE, "system", "NM-1", reason=reason)
        assert event.reason == reason


class TestQueries:
    def test_history_filters_by_target(self, log):
        log.record(CREATE, "system", "NM-1")
        log.record(CREATE, "system", "NM-2")
        assert len(log.history("NM-1")) == 1

    def test_by_actor(self, log):
        log.record(CREATE, "alice", "NM-1")
        log.record(CREATE, "bob", "NM-2")
        assert len(log.by_actor("alice")) == 1

    def test_by_action(self, log):
        log.record(CREATE, "system", "NM-1")
        log.record(MERGE, "system", "NM-1")
        assert len(log.by_action(MERGE)) == 1

    def test_frame_is_newest_first(self, log):
        log.record(CREATE, "system", "NM-1")
        log.record(CREATE, "system", "NM-2")
        assert log.to_frame().iloc[0]["target"] == "NM-2"


class TestRollback:
    def test_appends_rather_than_deletes(self, log):
        """History must survive its own correction."""
        log.record(CREATE, "system", "NM-1", after={"members": ["A", "B"]})
        log.rollback(0, "auditor")
        assert len(log.events) == 2
        assert log.events[0].action == CREATE

    def test_cross_links_both_directions(self, log):
        log.record(CREATE, "system", "NM-1")
        compensating = log.rollback(0, "auditor")
        assert compensating.compensates == 0
        assert log.events[0].reverted_by == compensating.event_id

    def test_swaps_before_and_after(self, log):
        log.record(CREATE, "system", "NM-1", before={"x": 1}, after={"x": 2})
        compensating = log.rollback(0, "auditor")
        assert compensating.before == {"x": 2}
        assert compensating.after == {"x": 1}

    def test_double_rollback_rejected(self, log):
        log.record(CREATE, "system", "NM-1")
        log.rollback(0, "auditor")
        with pytest.raises(ValueError, match="already reversed"):
            log.rollback(0, "auditor")

    def test_rolling_back_a_rollback_rejected(self, log):
        log.record(CREATE, "system", "NM-1")
        log.rollback(0, "auditor")
        with pytest.raises(ValueError, match="itself a rollback"):
            log.rollback(1, "auditor")

    def test_unknown_event_rejected(self, log):
        with pytest.raises(IndexError):
            log.rollback(99, "auditor")

    def test_cross_links_survive_reload(self, log):
        log.record(CREATE, "system", "NM-1")
        log.rollback(0, "auditor")
        assert AuditLog(path=log.path).events[0].reverted_by == 1

    def test_effective_events_excludes_reversed_pair(self, log):
        log.record(CREATE, "system", "NM-1")
        log.record(CREATE, "system", "NM-2")
        log.rollback(0, "auditor")
        targets = [e.target for e in log.effective_events()]
        assert targets == ["NM-2"]


class TestIntegrity:
    def test_clean_log_passes(self, log):
        log.record(CREATE, "system", "NM-1")
        log.record(MERGE, "system", "NM-1")
        assert log.integrity_check()["ok"]

    def test_rollbacks_counted(self, log):
        log.record(CREATE, "system", "NM-1")
        log.rollback(0, "auditor")
        assert log.integrity_check()["n_rollbacks"] == 1

    def test_broken_cross_link_detected(self, log):
        log.record(CREATE, "system", "NM-1")
        log.rollback(0, "auditor")
        log.events[0].reverted_by = None  # simulate tampering
        result = log.integrity_check()
        assert not result["ok"]
        assert any("cross-linked" in p for p in result["problems"])


class TestVersioning:
    def test_state_reconstructed_at_a_point_in_time(self, log):
        """Uses as_of_event_id to break the tie when both events land on the
        same timestamp -- routine on Windows, where clock resolution is
        coarser than the time between these two calls."""
        first = log.record(CREATE, "system", "NM-1", after={"members": ["A"]})
        log.record(CREATE, "system", "NM-2", after={"members": ["B"]})
        state = log.version_at(first.timestamp, as_of_event_id=first.event_id)
        assert "NM-1" in state
        assert "NM-2" not in state

    def test_without_as_of_event_id_ties_include_every_event_at_timestamp(self, log):
        """The tie-blind default is unchanged: a bare timestamp cutoff
        cannot distinguish same-timestamp events, so both are included."""
        first = log.record(CREATE, "system", "NM-1", after={"members": ["A"]})
        second = log.record(CREATE, "system", "NM-2", after={"members": ["B"]})
        second.timestamp = first.timestamp  # force the collision deterministically
        state = log.version_at(first.timestamp)
        assert "NM-1" in state
        assert "NM-2" in state

    def test_as_of_event_id_excludes_later_events_at_the_same_timestamp(self, log):
        first = log.record(CREATE, "system", "NM-1", after={"members": ["A"]})
        second = log.record(CREATE, "system", "NM-2", after={"members": ["B"]})
        second.timestamp = first.timestamp  # force the collision deterministically
        state = log.version_at(first.timestamp, as_of_event_id=first.event_id)
        assert "NM-1" in state
        assert "NM-2" not in state
