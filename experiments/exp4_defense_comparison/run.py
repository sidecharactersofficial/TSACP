"""Exp 4 placeholder — NOT IMPLEMENTED (partially reproducible elsewhere).

The AT/RS baselines require artifacts (an adversarially-trained checkpoint and
a randomized-smoothing certificate) that are not shipped. The reproducible part
of this comparison — DACM (static) vs TASCP naive vs TASCP adaptive vs
undefended — is implemented and archived by exp5 (see
experiments/exp5_adaptive_bpda_attack/ and
results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json).
"""
from experiments._not_implemented import not_implemented

if __name__ == "__main__":
    not_implemented(
        "exp4", "C1 (defense comparison)",
        "AT/RS baselines lack shipped checkpoints; the DACM/TASCP/undefended "
        "comparison is reproduced by exp5.",
    )
