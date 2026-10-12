# Architecture

## Two-Loop Design

TASCP uses a two-loop architecture:

```mermaid
graph TD
    subgraph Fast_Loop["Fast Loop (Rust)"]
        A1["1. Track query"]
        A2["2. Compute M_t"]
        A3["3. Compute θ"]
        A4["4. Rotate input"]
        A5["5. Classify"]
        A6["6. Log event"]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end
    Request["Request"] --> A1
    A5 --> Response["Response"]
    A6 -->|event stream| B1
    subgraph Slow_Loop["Slow Loop (Python)"]
        B1["1. Pattern analysis"]
        B2["2. Recalibrate γ, τ, ω, λ"]
        B1 --> B2
    end
    B2 -.->|update params| A3
```

- **Fast loop** (Rust, `tascp-proto/fast-loop/`): in-memory sliding-window
  history, per-request Givens rotation, synchronous, <1ms target.
- **Slow loop** (Python, `tascp-proto/slow-loop/`): async event-log analysis,
  recalibrates defense parameters, no latency constraint.

## Evaluation Framework

```mermaid
graph TD
    subgraph models["models/"]
        M1["nids_classifier.py<br/>3-layer MLP (PyTorch)"]
        M2["train.py<br/>UNSW-NB15 training"]
        M3["train_cicids2017.py<br/>CICIDS2017 training"]
        M4["xgboost_classifier.py<br/>XGBoost + surrogate MLP"]
        M5["eval_both.py<br/>Evaluate both datasets"]
        M6["weights/<br/>Trained model weights"]
    end
    subgraph defenses["defenses/"]
        D1["tascp.py<br/>TASCPWrapper"]
    end
    subgraph baselines["baselines/"]
        B1["dacm.py<br/>Static clamping"]
    end
    subgraph attacks["attacks/"]
        A1["pgd.py<br/>Standard PGD"]
        A2["bpda_adaptive.py<br/>Stateful PGD + BPDA"]
        A3["bpda_xgb_surrogate.py<br/>Surrogate-gradient BPDA"]
    end
    R["reference_math.py<br/>Momentum, penetration"] -.-> D1
    A2 -.-> D1
    A3 -.-> D1
```

## Defense Components (TASCPWrapper)

1. **Momentum tracking** — cosine similarity `M_t` between consecutive
   velocity vectors `v_t` and `v_{t-1}`.
2. **Penetration** — windowed average `eps_p = mean(||v_i|| * max(0, M_i))`.
3. **Rotation angle** — `delta_theta = gamma * clamp(eps_p - tau, 0)`.
4. **Givens rotation** — rotates in the span of `{u, v_t_perp}` where `u`
   is the normalized current velocity and `v_t_perp` is the orthogonal
   component of the previous velocity.

## Attack Components

1. **PGD** — standard multi-step projected gradient descent with L2/Linf
   norms and optional momentum (MIM).
2. **BPDA (stateful)** — PGD that interfaces with TASCPWrapper's history;
   supports adaptive=False (detached rotation) and adaptive=True (full
   gradient flow through rotation).
3. **Surrogate-gradient BPDA** — for non-differentiable targets: gradients
   from a surrogate MLP, success evaluation against the real model.

## Config System

Hydra YAML configs in `configs/`:
- `attack/pgd.yaml` — epsilon, alpha, iterations, norm
- `dataset/unsw_nb15.yaml`, `dataset/cicids2017.yaml`
- `base.yaml` — shared defaults
