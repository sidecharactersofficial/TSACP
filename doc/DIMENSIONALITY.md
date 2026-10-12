# Feature dimensionality across datasets

This note reconciles the `d` value quoted in the paper with the value the
shipped artifact actually uses, so a reviewer can verify both.

| Dataset | Paper `d` | Artifact `d_feature` | Why |
| :--- | :---: | :---: | :--- |
| UNSW-NB15 | **49** | **43** | 49 raw columns; 6 non-model columns are dropped: the 5-tuple session keys (`srcIP`, `srcport`, `dstIP`, `dstport`, `proto`) used only to form sessions, plus the label column. 49 − 6 = 43. |
| CICIDS2017 | — | **78** | CICIDS2017 schema after label/timestamp handling. |
| NF-ToN-IoT-v2 | — | **43** | NetFlow-v9 numeric features after dropping the src/dst IPv4 address columns and the `Label`/`Attack` identity columns. |

## Reconciliation

The paper reports `d = 49` for UNSW-NB15 because that is the dimensionality of
the raw feature table **including** the 5-tuple identity columns. Those columns
are categorical identifiers, not model inputs: `data/scripts/preprocess.py`
uses them only to group rows into sessions and then drops them before fitting
the classifier. Every experiment therefore operates in `d = 43`.

Both numbers are correct for their respective objects; the artifact records the
`d = 43` value in `configs/dataset/unsw_nb15.yaml` and everywhere in
`src/tascp_eval/`. The formal results in `doc/THEOREMS.md` that reference `d`
are generic in `d` and hold for either value.

Verify:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/processed/unsw_nb15/unsw_nb15_processed.parquet")
print("columns:", len(df.columns), "| model features:", len(df.columns) - 2)  # minus label + session_id
PY
```
