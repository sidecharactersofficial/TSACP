# NF-ToN-IoT-v2 — data provenance

- Source: HuggingFace dataset `Nora9029/NF-ToN-IoT-v2`, file
  `NF-ToN-IoT-v2-test.csv` (531,923,548 bytes, downloaded 2026-10-08 via
  https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2/resolve/main/NF-ToN-IoT-v2-test.csv).
- Original upstream: the NF-ToN-IoT-v2 NetFlow dataset (ToN-IoT telemetry
  flows in NetFlow-v9 format).
- Local raw copy: `data/raw/ton_iot/NF-ToN-IoT-v2-test.csv` (508 MB,
  gitignored — raw data is not committed).
- Processed: `data/processed/ton_iot/ton_iot_processed.parquet`
  (3,388,100 rows, 41 float32 features + binary label, 64.0% attacks).
  Columns `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `Attack` dropped (address
  strings / family string; the binary pipeline uses `Label`).
  Normalization: sign-preserving log1p then per-feature min-max to [0,1]
  (stats computed over the full file in a first streaming pass).
- Trained classifier: `src/tascp_eval/models/weights/ton_iot_nids.pth`
  (NIDSClassifier 2x64, same protocol as the UNSW/CICIDS models).
- Reproduce: `python data/ton_iot/prepare_ton_iot.py && python
  data/ton_iot/train_ton_iot.py`.
