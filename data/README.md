# Data Directory

This directory contains the datasets used for empirical evaluation of TASCP.

## Datasets
1. **UNSW-NB15**
   - **Source**: Kaggle mirror `mrwellsdavid/unsw-nb15` (Standard CSV release).
   - **Date Downloaded**: July 5, 2026.
   - **Preprocessing**: Categorical features (`proto`, `service`, `state`) are factorized. The sequences are grouped into directionless sessions using `(srcip, dstip, sport, dsport, proto)`. Features are MinMax scaled. Dimensionality is 49.

2. **CICIDS2017**
   - **Source**: Kaggle mirror `shadman1028/cicids2017-official-flow-feature-csv-files` (Official Flow Feature CSVs).
   - **Date Downloaded**: July 5, 2026.
   - **Preprocessing**: Handles raw CSVs by stripping whitespace from columns. Replaces `Infinity` and `NaN` (often in `Flow Bytes/s` or `Flow Packets/s`) with the feature median to prevent gradient explosion during attacks. Bidirectional flows are merged into directionless trajectory sessions via a sorted mapping of `(src_ip, dst_ip, src_port, dst_port, protocol)`. Features are MinMax scaled. Dimensionality is 78.

## Archived Data
The legacy NSL-KDD placeholder data previously used during prototype scaffolding is preserved in `raw/_archived_nsl_kdd/`. Do not use it for empirical evaluations.
