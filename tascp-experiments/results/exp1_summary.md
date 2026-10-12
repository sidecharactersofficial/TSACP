# Experiment 1: Momentum Separation (UNSW-NB15)

## Overview
Experiment 1 evaluates the premise of Lemma 2: that malicious attacks exhibit greater structural momentum (higher step-to-step consistency) within a session trajectory than benign background traffic. We define momentum $M_t$ at step $t$ as the cosine similarity between the current and previous feature vectors within the same session ($M_t = \cos(\angle(v_t, v_{t-1}))$).

## Methodology
- **Dataset**: UNSW-NB15 (Standard 4-part CSV release)
- **Dimensionality**: $d=43$ (49 raw features minus 6 key/label columns dropped to prevent overfitting).
- **Session Grouping**: Directionless 5-tuple (Src IP, Dst IP, Src Port, Dst Port, Protocol).
- **Measurement**: We extracted all sessions with $|W| > 1$. For each step $t \ge 1$ within a session, we computed $M_t$.
- **Validation Metric**: We compared the distribution of $M_t$ for Benign vs. Attack steps and computed the Area Under the Receiver Operating Characteristic Curve (AUC) using $M_t$ as a simple threshold classifier.

## Results
A total of 337,354 valid intra-session steps were measured (139,331 Benign, 198,023 Attack).

### Summary Statistics
| Class | Mean $M_t$ | Variance |
|---|---|---|
| Benign | 0.1118 | 0.0980 |
| Attack | 0.0450 | 0.0428 |

### AUC
The Area Under the ROC Curve (AUC) for separating Attack from Benign using $M_t$ alone as a threshold is **0.4658**.

## Observations

### Section A: Historical attack traffic vs benign (original exp1)
This section tests whether attack-labeled traffic in general has elevated momentum. 
**Result:** No (AUC 0.4658).
The measured mean $M_t$ for Attack traffic (0.0450) is lower than the mean $M_t$ for Benign traffic (0.1118). Because the attack class exhibits lower cosine similarity between steps than the benign class, a simple threshold classifier predicting "Attack" when $M_t$ is high yields an AUC slightly below 0.5 (0.4658). These numbers reflect the empirical measurements of the standard dataset prior to any adaptive adversarial perturbations.

### Section B: Live PGD-trajectory vs benign (exp1b)
This section tests the paper's actual claim about gradient-based optimizers (Assumption 2): that a live optimization trajectory exhibits higher structural momentum than natural traffic. We trained a 3-layer NIDS classifier (F1: 0.9631, Accuracy: 99.05%) on the UNSW-NB15 features and evaluated standard white-box PGD (L2 norm) trajectories over 2000 correctly classified benign points.
**Result:** Yes (AUC 0.9053).
The mean $M_t$ across live PGD trajectories is **1.0000** (Var: 0.0000), compared to the natural benign traffic mean of **0.1118**. Because adversarial trajectories take iterative, consistent gradient steps (causing consecutive feature vectors to be highly aligned), thresholding on $M_t$ successfully separates a live gradient-based attack from historical benign traffic with high confidence (AUC 0.9053).

A visual histogram showing the distributions of historical $M_t$ across classes has been saved to `results/figures/exp1_momentum_distribution.png`.
