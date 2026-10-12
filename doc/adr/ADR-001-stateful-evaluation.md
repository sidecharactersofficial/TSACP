# ADR-001: Stateful Evaluation and Query-Fed Defense History

## Status

Accepted.

## Context

TASCP is a *stateful* defense: its per-request behavior depends on the
trajectory of previous requests in the current session (a rolling window used
to compute momentum, penetration, and the rotation angle). Evaluating it as a
stateless function would misrepresent both its intended behavior and its
failure modes.

Adversaries in the white-box setting can feed arbitrary queries into the
defense; if history were not updated per query, an attacker could effectively
reset the defense between steps.

## Decision

All evaluation drivers update the defense's history on **every distinct query**
the attacker submits, not only on the final candidate. The helper
`src/tascp_eval/attacks/stateful.py` (`init_stateful_history`, `observe_query`)
is the single interface for this.

Sessions are tracked per client identity. Offline attackers that submit a
single fresh flow therefore see the identity-transform branch
(`Δθ = 0` on the first query of a session).

## Consequences

### Positive
- Faithful to the defense's design and to the attacker's query access.
- Makes the black-box trigger blindness (Corollary 3) measurable: NES's
  antithetic queries drive momentum negative, so the rotation never fires.
- Makes the offline bit-identity result (exp15) well-defined.

### Negative
- Results depend on session boundaries and query ordering; a stateless
  reimplementation would not reproduce them.
- Per-session detection improvements only hold per identity; identity rotation
  defeats the per-session aggregator (exp11).

### Neutral
- `tests/test_blackbox_attacks.py::test_stateful_wrapper_sees_every_query`
  enforces the query-feeding contract.
