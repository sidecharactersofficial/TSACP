# Exp 23: Audit of Existing Stateful Defenses (RQ6 / Tier-3)

## Question
Can the three pre-deployment audits (coupling, trigger observability,
stationarity) distinguish different uses of query-history state —
perturbation (TASCP) vs detection/rejection?

## Method
Three stateful/query-history defenses, identical audit protocol:
  tascp            trajectory-conditioned transformation (this paper)
  blacklight_sim   Blacklight-STYLE near-duplicate density detector
                   (exp10's L2 simplification of Li et al. USENIX Sec'22;
                   tau_match=0.1, k_min=5, window=50 — flagged queries are
                   rejected fail-closed)
  rate_limit       per-session query-budget limiter (budget=50)

Audits:
  A1 coupling: C_norm via autograd (tascp only — the others have no
     differentiable transform); empirical analog for all: nu_allow.
  A2 trigger observability: for tascp, attack SUCCESS rate through the
     stateful defense (evasion = still succeeding); for detectors, the
     fraction of attack sessions completing with ZERO alerts. Attacks:
     NES-independent, antithetic NES jitter, Square, random search
     (max 100 queries), n=100 benign points.
  A3 stationarity: frozen-probe replay with a persistent-direction session
     stream; probe pool MIXES benign flows and PGD-adversarial points so
     the toward-allow flip rate (nu_allow) is measured on the v0=1 subset.

## Results — UNSW-NB15
| defense       | coupling | nu_any | nu_allow | undetected/success (NES/jitter/Square/rand) | benign FP |
|---------------|----------|--------|----------|---------------------------------------------|-----------|
| tascp         | 8.5e7    | 14.86% | **36.22%** | 0% / 0% / 0% / 99%                          | 0%        |
| blacklight_sim| n/a      | 43.86% | 0%       | 0% / 0% / 0% / 0%                           | 100%      |
| rate_limit    | n/a      | 0%     | 0%       | 0% / 0% / 0% / 0%                           | 0%        |

## Results — NF-ToN-IoT-v2
| defense       | coupling | nu_any | nu_allow | undetected/success (NES/jitter/Square/rand) | benign FP |
|---------------|----------|--------|----------|---------------------------------------------|-----------|
| tascp         | 2.1e7    | 28.30% | **18.49%** | 3% / 13% / 70% / 58%                        | 0%        |
| blacklight_sim| n/a      | 35.65% | 0%       | 0% / 0% / 0% / 0%                           | 100%      |
| rate_limit    | n/a      | 0%     | 0%       | 0% / 0% / 0% / 0%                           | 0%        |

## Findings (honest)
1. **The audits separate the defense classes.** TASCP: nonzero coupling,
   non-stationary verdicts in BOTH directions (nu_allow 36.2% UNSW /
   18.5% TON-IoT — adversarial verdicts flip back to benign as session
   state evolves, the exploitable direction). Detectors/rate-limiter:
   no differentiable coupling channel; verdict changes are monotone
   toward reject (nu_allow = 0 exactly, both datasets).
2. **Outcome table** (the reviewer's killer table):
   - tascp: coupling YES, trigger blind spot YES (random search 99%/58%
     undetected-equivalent), non-stationary YES -> unsafe perturbation.
   - blacklight_sim: coupling NO, trigger blind spot NO at these settings
     (all attack sessions flagged), non-stationary but MONOTONE -> safe
     direction, with a real benign-traffic cost (below).
   - rate_limit: coupling NO, trigger NO (budget breached by every attack
     session at 100 queries vs budget 50), stationary -> safe but purely
     because the budget is below the attack session length; any
     legitimate session > 50 queries is also flagged.
3. **Honest calibration caveat**: blacklight_sim flags 100% of benign
   re-check sessions (40 near-identical re-checks of one flow with 1e-3
   jitter — k_min=5 hits within the 50-query window). This is the known
   operational tradeoff of near-duplicate detection: it is monotone-safe
   precisely because it treats benign repetition as attack-shaped. The
   detector's protection claim stands only with this benign-traffic cost
   on the table; exp10/exp11 report the complementary calibration with
   distinct-query benign sessions.
4. Claim discipline (per the reviewer's warning): this is NOT "all
   moving-target defenses fail". The supported claim: **trajectory-
   conditioned perturbation fails the audits; stateful detection/rejection
   passes the direction-of-flip audit at the price of benign-traffic FP.**

## Artifacts
- results/tables/exp23_defense_audit_{unsw_nb15,ton_iot}.csv
- results/logs/exp23_defense_audit/metrics_{dataset}.json
- results/logs/runs_session5/exp23_{unsw,toniot}.log
