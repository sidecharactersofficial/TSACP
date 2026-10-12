# Exp 19: Trajectory-Conditioned Transformation Family (RQ1 + RQ4)

## Question
Is the exploitable state channel specific to TASCP's spiral/rotation
construction, or does ANY trajectory-conditioned transformation
`x_t -> h_t -> T(h_t, x_t) -> f(T(h_t, x_t))` expose it?

## Method
Five controlled synthetic members of the same design family
(`experiments/exp19_trajectory_family/family.py`), sharing the state
extraction stage (windowed directional momentum) and differing ONLY in the
transformation: tascp (rotation, the original), tscale (feature-wise
scaling), tshift (translation along running mean direction), tshear
(rank-1 shear), tnoise (state-scaled reparameterized noise). Same protocol
as exp5: naive straight-through PGD, adaptive B2 (gradient through a local
lockstep replica), fixed-state control (S clamped to 0 -> identity
transform), plus coupling C = ||dT/dh||_F on an active trajectory and the
stationarity replay (nu_session / nu_final). PGD: eps 0.3, alpha 0.01,
20 iters, L2. n=500 benign points per dataset.

## Results — UNSW-NB15
| variant  | naive | adaptive | fixed-state | dASR (adaptive) | C        | nu_session |
|----------|-------|----------|-------------|-----------------|----------|------------|
| tascp    | 82.8% | 67.8%    | 57.4%       | +10.4pp         | 7.3e7    | 5.73%      |
| tscale   | 54.0% | 62.2%    | 57.4%       | +4.8pp          | 1.36     | 0.18%      |
| tshift   | 100.0%| 99.8%    | 57.4%       | +42.4pp         | 4.00     | 10.42%     |
| tshear   | 37.4% | 8.6%     | 57.4%       | -48.8pp         | 4.8e6    | 0.02%      |
| tnoise   | 56.6% | 58.2%    | 57.4%       | +0.8pp          | 0.35     | 0.20%      |

## Results — NF-ToN-IoT-v2 (third dataset, see exp24)
| variant  | naive | adaptive | fixed-state | dASR (adaptive) | C        | nu_session |
|----------|-------|----------|-------------|-----------------|----------|------------|
| tascp    |  —    |   —      |    —        | +4.8pp          | 4.1e7    | 37.02%     |
| tscale   |  —    |   —      |    —        | +0.4pp          | 1.96     | 8.32%      |
| tshift   |  —    |   —      |    —        | +10.6pp         | 3.66     | 15.73%     |
| tshear   |  —    |   —      |    —        | -2.2pp          | 5.5e6    | 8.53%      |
| tnoise   |  —    |   —      |    —        | +0.0pp          | 0.32     | 16.07%     |

(TON-IoT per-condition ASR columns are in
`results/tables/exp19_family_ton_iot.csv`; the dASR column is the headline.)

## Findings (honest)
1. **The phenomenon is NOT TASCP-specific.** Three of five transformations
   (rotation, scaling, translation) show a significant adaptive-vs-fixed-state
   gap on UNSW (2/5 on TON-IoT); the channel opens in every construction
   where state modulates the geometry of the decision boundary.
2. **Variation, not uniform failure** — exactly what the reviewer asked for.
   tshift AMPLIFIES the attacker (+42.4pp over fixed-state on UNSW);
   tshear SUPPRESSES it (-48.8pp: the shear moves benign-space attack
   candidates away from the boundary). tnoise is inert (its state gate
   saturates without geometric correlation).
3. **Stationarity violations are family-wide**: 2/5 variants violate
   nu_session > 1% on UNSW; **5/5 on TON-IoT** (37.0% for TASCP, 16.1%
   even for tnoise). Non-stationarity does not require rotation.
4. Caveat: on UNSW the adaptive (replica) attack is weaker than the naive
   straight-through attack for tascp (67.8% vs 82.8%) — the lockstep replica
   is an approximation of the deployed history; we report both.

## Scope note
These are controlled synthetic members of the same design family, NOT
published defenses; the claim is about the design pattern, not about any
prior system.

## Artifacts
- results/tables/exp19_family_{unsw_nb15,ton_iot}.csv
- results/logs/exp19_trajectory_family/metrics_{dataset}.json
- results/logs/runs_session5/exp19_{unsw,toniot}.log
