"""Shared helpers for attacks that must drive a stateful defense's query history.

Stateful defenses (TASCP) observe every query a client issues. The attack loop
is responsible for feeding each *distinct* query point into the wrapper's
history so the defense sees exactly what a real client session looks like.

Convention (mirrors `bpda_adaptive.stateful_pgd_attack`):
  - Before the first forward:  reset_history(batch, device); add_to_history(x0)
  - After each forward of q_i: add_to_history(q_i)

so that inside `forward(x_t)` the history's last entry is always the
*previous* distinct query (v_t = x_t - x_{t-1} is well-defined).
"""


def init_stateful_history(model, x):
    """Prepare a stateful wrapper to observe a fresh session rooted at x."""
    if hasattr(model, "reset_history"):
        model.reset_history(x.size(0), x.device)
        model.add_to_history(x.clone().detach())


def observe_query(model, x_q):
    """Record a query the defender has just answered."""
    if hasattr(model, "add_to_history"):
        model.add_to_history(x_q.clone().detach())
