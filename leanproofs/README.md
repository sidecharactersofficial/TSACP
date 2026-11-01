# Project Archimedes (Singularity): Machine-Checked Formal Verification in Lean 4

This directory contains the machine-checked formal verification in Lean 4 and the step-by-step mathematical soundness verification for all theoretical foundations, bounds, lemmas, propositions, and corollaries established in the paper:
**"Singularity: The Illusion of Dynamic Security"** (*IEEE Symposium on Security and Privacy*).

---

## Directory Structure

- `MATH_AUDIT_REPORT.md`: Comprehensive line-by-line mathematical audit report dissecting every lemma, proposition, theorem, and corollary in `paper/main (8).tex` and Appendix A (`\section{Proofs}`).
- `verify_math.py`: Standalone executable Python mathematical verification suite checking algebraic identities, Isserlis Gaussian moment expansions, Laurent-Massart concentration bounds, and Monte Carlo empirical validations.
- `ArchimedesProofs/Basic.lean`: Lean 4 formal proofs for all 11 core mathematical theorems, lemmas, propositions, and corollaries (compiled with **0 `sorry`**).
- `ArchimedesProofs.lean`: Root library entry point for Lake.
- `Main.lean`: Executable driver reporting the verification status.
- `lakefile.toml` / `lean-toolchain`: Lake project configuration (Lean v4.35.0).
- `.github/workflows/lean_action_ci.yml`: Continuous formal verification workflow via GitHub Actions (`leanprover/lean-action@v1`).

---

## Verified Theorems, Lemmas, Propositions, and Corollaries

| Paper Section | Label | Description | Lean 4 Formalization | Status |
|:---|:---|:---|:---|:---:|
| **§3.3** | `prop:obs` | **Proposition 1**: Observability Gap & Single-Query Collapse | `prop1_schedule_evaluated_zero`<br>`prop1_schedule_prototype_zero`<br>`prop1_rotation_identity_at_zero`<br>`prop1_observability_gap` | **PROVED** |
| **§4.3 & App. A.1** | `lem:struct` | **Lemma 1**: Structure of Givens Rotation & Projection Equivariance | `lem1_givens_in_plane_action`<br>`lem1_givens_orthogonal_action`<br>`lem1_givens_orthogonality_algebra`<br>`lem1_projection_equivariance` | **PROVED** |
| **§4.4 & App. A.2** | `lem:decay` | **Lemma 2**: Gradient Correlation Decay & Isotropic Invariant | `lem2_gradient_correlation_algebra`<br>`lem2_in_plane_reduction`<br>`lem2_isotropic_d43_expectation` | **PROVED** |
| **§4.4 & App. A.3** | `lem:angle` | **Lemma 3**: Angle-Channel Gradient & Product Rule Cancellation | `lem3_momentum_product_rule_cancellation`<br>`lem3_clipped_gradient_zero`<br>`lem3_angle_gradient_rank_one` | **PROVED** |
| **§4.4 & App. A.4** | `prop:query-bound` | **Proposition 2**: One-Step Angle Bound Below Threshold | `prop2_angle_bound_evaluated`<br>`prop2_angle_bound_prototype`<br>`prop2_cosine_monotonicity` | **PROVED** |
| **§4.5 & App. A.4** | `thm:leak` | **Theorem 1**: No Obfuscation Barrier & Full Gradient Reconstruction | `thm1_differential_decomposition`<br>`thm1_ablation_basis_zero`<br>`thm1_ablation_angle_zero`<br>`thm1_basis_norm_scaling` | **PROVED** |
| **§4.6 & App. A.5** | `prop:qc` | **Proposition 3**: Query Tax of Gradient Estimation (NES Scaling) | `prop3_first_moment_expectation`<br>`prop3_second_moment_expansion`<br>`prop3_ratio_of_expectations`<br>`prop3_delta_denominator_identity` | **PROVED** |
| **§4.6 & App. A.6** | `cor:blind` | **Corollary 1**: Antithetic Blindness & Exponential Tail Invariance | `cor1_nes_steps_algebra`<br>`cor1_nes_inner_product_algebra`<br>`cor1_positivity_condition`<br>`cor1_nes_asymptotic_limit`<br>`cor1_d43_exponential_tail_bound` | **PROVED** |
| **§4.7 & App. A.7** | `prop:domain` | **Proposition 4**: Domain Constraints Shrink Feasible Set | `prop4_feasible_set_shrinkage`<br>`prop4_heuristic_lower_bound` | **PROVED** |
| **§4.8 & App. A.8** | `prop:mono` | **Proposition 5**: Rejection-Only Mechanisms are Directionally Safe | `prop5_rejection_only_directional_safety`<br>`prop5_monotone_flag_no_resurrection` | **PROVED** |
| **§4.8 & App. A.9** | `prop:disp` | **Proposition 6**: Where Verdicts Can Flip (Decision Boundary Tube Bound) | `prop6_rotation_displacement_norm_algebra`<br>`prop6_triangle_inequality_tube_bound`<br>`prop6_boundary_flip_necessary_condition` | **PROVED** |

---

## How to Run the Verification

### 1. Build and Run the Lean 4 Formal Proofs
```bash
cd leanproofs
lake build
./.lake/build/bin/archimedes_proofs
```

### 2. Check Individual File with Lean 4
```bash
lean leanproofs/ArchimedesProofs/Basic.lean
```

### 3. Run the Computational and Empirical Verification Suite
```bash
python3 leanproofs/verify_math.py
```

