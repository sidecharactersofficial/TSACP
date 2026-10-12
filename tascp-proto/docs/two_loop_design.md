# TASCP Two-Loop Architecture Design

The TASCP prototype employs a strict two-loop architecture:

## 1. Fast Loop (In-path)
- **Budget**: Sub-millisecond.
- **Responsibility**: Maintain short-term momentum window, compute penetration/rotation, and apply algebraic constraints.
- **State**: Ephemeral, in-memory (`DashMap`, `ArcSwap`), per-session.
- **IPC**: Reads parameters (`gamma`, `lambda`, `k`, `tau`) from a lock-free structure updated by the slow loop.

## 2. Slow Loop (Out-of-path)
- **Budget**: Lag-tolerant (seconds/minutes).
- **Responsibility**: Asynchronously read event store (SQLite), compute attack-family signatures over long horizons, classify, and write updated parameters back to the fast loop via the HTTP `/admin/params` endpoint.
- **Constraint**: Must NEVER block the fast path. If the slow loop lags, the fast loop operates on slightly stale, safe default parameters.

## Tradeoffs and Limitations
- The slow loop does NOT make real-time decisions. It calibrates.
- No real-time attack-family detection within the request latency budget.
- We do not claim white-box immunity here; refer to the paper scope for theoretical threat models.
