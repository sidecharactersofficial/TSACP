"""Prepare NF-ToN-IoT-v2 (third dataset, RQ5/D) into the repo's processed format.

Source: Nora9029/NF-ToN-IoT-v2 (HuggingFace), test split - 43 NetFlow-v9
numeric features + IPV4_SRC_ADDR / IPV4_DST_ADDR (dropped: raw address
strings, not flow semantics; consistent with the other pipelines'
handling of non-numeric identity columns) + Label (binary) + Attack
(family string, kept for stratification records but not used by the
binary pipeline).

This is a genuinely DIFFERENT tabular security dataset from UNSW-NB15 /
CICIDS2017: different feature schema (NetFlow-v9 instead of packet/flow
hybrid), different class distribution (IoT attack traffic), and a
different flow-generation process (IoT telemetry).

Output: data/processed/ton_iot/ton_iot_processed.parquet with columns
[<43 feature cols float32>, label], benign=0/attack=1 (same convention
as the other processed sets). Streaming read; peak memory = one chunk.
"""

import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(PROJECT_ROOT, "data", "raw", "ton_iot", "NF-ToN-IoT-v2-test.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ton_iot")
OUT = os.path.join(OUT_DIR, "ton_iot_processed.parquet")

CHUNK = 500_000
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"]


def main():
    if not os.path.exists(RAW):
        raise FileNotFoundError(
            f"{RAW} not found. Download NF-ToN-IoT-v2-test.csv from "
            "https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2 first."
        )
    os.makedirs(OUT_DIR, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    header = pd.read_csv(RAW, nrows=0)
    feature_cols = [c for c in header.columns if c not in DROP_COLS + ["Label"]]
    n_feat = len(feature_cols)

    # Pass 1: per-feature compression + range stats. Raw NetFlow features span
    # 1e-3..1e7 (bytes, durations, throughputs) and blow up MLP training
    # without scaling; the repo's other pipelines operate in [0,1] (attacks
    # clamp to [0,1]), so we apply sign-preserving log1p then min-max to [0,1].
    gmin = np.full(n_feat, np.inf, dtype=np.float64)
    gmax = np.full(n_feat, -np.inf, dtype=np.float64)
    for chunk in pd.read_csv(RAW, chunksize=CHUNK, usecols=feature_cols):
        Xc = chunk.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        Xc = np.log1p(np.abs(Xc)) * np.sign(Xc)
        gmin = np.minimum(gmin, Xc.min(axis=0))
        gmax = np.maximum(gmax, Xc.max(axis=0))
        del chunk, Xc
    gmax = np.where(gmax - gmin < 1e-12, gmin + 1.0, gmax)  # constant cols -> 0
    print("log1p ranges computed")

    writer = None
    total = pos = 0
    first_chunk = True
    for chunk in pd.read_csv(RAW, chunksize=CHUNK):
        if first_chunk:
            chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns])
        y = chunk["Label"].astype(np.int64).values
        Xc = chunk[feature_cols].apply(pd.to_numeric, errors="coerce")
        Xc = Xc.fillna(0.0).to_numpy(dtype=np.float64)
        Xc = np.log1p(np.abs(Xc)) * np.sign(Xc)
        Xc = np.clip((Xc - gmin) / (gmax - gmin), 0.0, 1.0).astype(np.float32)
        d = pd.DataFrame(Xc, columns=feature_cols)
        d["label"] = y
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT, tbl.schema)
        writer.write_table(tbl)
        total += len(d)
        pos += int(y.sum())
        first_chunk = False
        del chunk, Xc, y, d, tbl
    if writer:
        writer.close()
    print(f"wrote {OUT}")
    print(f"rows={total} attacks={pos} ({pos/total*100:.1f}%)")
    print(f"features={n_feat}")


if __name__ == "__main__":
    main()
