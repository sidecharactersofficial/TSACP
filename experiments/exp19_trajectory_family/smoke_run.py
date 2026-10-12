"""End-to-end synthetic smoke test for exp19 eval_variants (all paths)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()) if False else ".", ""))
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from experiments.exp19_trajectory_family.run import eval_variants, PGD_CFG

torch.manual_seed(0)
d = 12
X = torch.rand(32, d)
y = ((X[:, :4].sum(1, keepdim=True) < 2.0).float())
base = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))

cfg = dict(PGD_CFG)
cfg["iterations"] = 6  # fast

results = eval_variants(base, X, y, cfg)
assert set(results) == {"tascp", "tscale", "tshift", "tshear", "tnoise"}
for v, r in results.items():
    for k in ("naive_asr", "adaptive_asr", "fixed_state_asr", "coupling_C",
              "coupling_firing_frac", "nu_session", "nu_final"):
        assert isinstance(r[k], float) and (0.0 <= r[k] <= 1.0 or k == "coupling_C" or r[k] >= 0.0), (v, k, r[k])
print("\nSMOKE OK")
