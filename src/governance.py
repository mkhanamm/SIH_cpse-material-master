"""
Audit trail, versioning and rollback (Spec section 4.10).

WHAT THIS FILE DOES
    Makes every change to the national mapping reconstructible and reversible.
    Nothing in this system mutates the mapping table silently.

    Every create / update / merge / split / approve / reject writes an
    append-only event: event id, ISO-8601 UTC timestamp, actor, action, target
    CNMC, before-state, after-state, and the reason (which for automated actions
    is the rendered Explanation -- so the audit log answers "why did the machine
    do this?" not just "what changed?").

    Rollback replays the log to a prior version rather than deleting history:
    an incorrect auto-approval is undone by appending a compensating event, so
    the record that it happened survives. This is what makes the system
    auditable by a CAG-style reviewer.

INPUTS
    Events emitted by cnmc_generator, review_workflow and matching_engine.

OUTPUTS
    outputs/audit_log.jsonl - append-only event log
    AuditLog                - queryable view (by CNMC, actor, date, action)

KEY FUNCTIONS
    AuditLog.record(action, actor, target, before, after, reason) -> Event
    AuditLog.history(cnmc)        -> list[Event]
    AuditLog.rollback(event_id)   -> Event  (compensating event)
    AuditLog.version_at(timestamp)-> dict   (mapping state as of a point in time)
"""
