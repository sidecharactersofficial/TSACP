# Exp 21: Trigger x Attack-Family Matrix (RQ3)

## Question
Is the B1 trigger-blindness result (antithetic NES evades TASCP's momentum
trigger) specific to the momentum trigger? Reviewer: "that's just your
momentum trigger."

## Method
Five trajectory triggers evaluated on every recorded query stream
(sliding window W=5, matching TASCP):

  momentum          TASCP's windowed ||v||*max(0, cos) statistic
  cosine            mean signed cosine of consecutive displacements
  sign_agreement    fraction of coordinates whose displacement sign repeats
  displacement      directional persistence ||mean v|| / mean ||v||
  antithetic_pair   max over window of 1 - ||v_t + v_{t-1}||/(||v_t||+||v_{t-1}||)
                    (near-opposite equal-magnitude probe pairs)

Eight attack families run against the UNDEFENDED base model with a
query-recording wrapper: NES-independent, SPSA, ZOO, SimBA, Square,
Boundary, random search, and the B1 antithetic NES jitter variant.
Thresholds calibrated per trigger on a benign control session (60
re-checks of the same benign flow with 1e-3 jitter) so that NO benign
window statistic is exceeded — i.e. an attack fires only where its
statistic exceeds every benign window observed. n=100 benign points,
max 150 queries per attack session.

## Results — UNSW-NB15 (session fire rates)
| attack               | momentum | cosine | sign_agr | displac | antithetic |
|----------------------|----------|--------|----------|---------|------------|
| nes_independent      | 100%     | 1%     | 2%       | 0%      | 0%         |
| spsa                 | 100%     | 0%     | 0%       | 8%      | 100%       |
| zoo                  | 35%      | 0%     | 100%     | 100%    | 0%         |
| simba                | 100%     | 0%     | 0%       | 2%      | 100%       |
| square               | 62%      | 100%   | 100%     | 100%    | 100%       |
| boundary             | 77%      | 99%    | 99%      | 46%     | 64%        |
| random_search        | 100%     | 100%   | 100%     | 100%    | 0%         |
| nes_antithetic_jitter| 93%      | 94%    | 0%       | 0%      | 1%         |

## Results — NF-ToN-IoT-v2 (highlights)
- momentum fires 87-100% for every attack family on this dataset (the
  IoT flow-generation process makes the momentum statistic separable);
- cosine / sign_agreement still collapse on specific families (1%/0% for
  SimBA; 1% for antithetic jitter), i.e. family-dependence persists.
Full matrix: `results/tables/exp21_trigger_matrix_ton_iot.csv`.

## Findings (honest)
1. **Trajectory statistics are attack-family dependent.** No trigger fires
   on >50% of sessions for every attack family on UNSW. Each trigger is
   blind to at least one family at a benign-calibrated threshold:
   - momentum: blind to ZOO (35%)
   - cosine: blind to NES/SPSA/ZOO/SimBA (0-1%)
   - sign_agreement: blind to SPSA/SimBA/antithetic jitter (0%)
   - displacement: blind to NES/jitter (0%)
   - antithetic_pair: blind to ZOO/random search (0%)
2. **Calibration cost is real**: at per-window 99% quantile thresholds the
   benign FP was 2-25% (heavy-tailed benign re-check traffic); raising to
   the per-sample benign maximum removes benign FPs but weakens detection
   of the boundary family (77-93% for momentum). Detection strength and
   benign FP trade off along the attack-family axis, not just the budget.
3. **Design principle (the paper's general claim)**: a stateful trigger
   must be evaluated against the query-generation statistics of the
   attacker, not merely against the designer's intuitive notion of
   "persistent movement". The B1 result is a property of the
   trigger/statistic pairing, not of one lucky attack.

## Artifacts
- results/tables/exp21_trigger_matrix_{unsw_nb15,ton_iot}.csv
- results/logs/exp21_trigger_matrix/metrics_{dataset}.json
- results/logs/runs_session5/exp21_{unsw,toniot}.log
