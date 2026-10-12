#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

def preprocess_unsw():
    print("--- Preprocessing UNSW-NB15 ---")
    raw_dir = "data/raw/unsw_nb15"
    out_dir = "data/processed/unsw_nb15"
    os.makedirs(out_dir, exist_ok=True)
    
    # Only use the raw files 1-4 which have the IP and port columns
    files = glob.glob(f"{raw_dir}/UNSW-NB15_[1-4].csv")
    if not files:
        print(f"No CSVs found in {raw_dir}")
        return
        
    dfs = []
    
    columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
    
    for f in files:
        df = pd.read_csv(f, encoding='latin1', low_memory=False, on_bad_lines='skip', header=None, names=columns)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded total {len(df)} rows.")
    
    # Label is usually 'label'
    if 'label' in df.columns:
        label_col = 'label'
    elif 'Label' in df.columns:
        label_col = 'Label'
    else:
        print("Error: Label column not found.")
        return

    # Categorical encoding (Target encoding for simplicity or just factorize for prototype)
    cat_cols = ['proto', 'service', 'state']
    for c in cat_cols:
        if c in df.columns:
            df[c] = pd.factorize(df[c])[0]
            
    # Drop non-numeric or ID columns
    drop_cols = ['id', 'attack_cat']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    # Scale numeric features
    numeric_cols = [c for c in df.columns if c != label_col and c not in ['srcip', 'sport', 'dstip', 'dsport']]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().any():
            df[c].fillna(df[c].median(), inplace=True)
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Group into sessions: (srcip, sport, dstip, dsport, proto)
    if all(c in df.columns for c in ['srcip', 'sport', 'dstip', 'dsport', 'proto']):
        # Create directionless tuple mapping
        def get_directionless(row):
            ips = sorted([str(row['srcip']), str(row['dstip'])])
            ports = sorted([str(row['sport']), str(row['dsport'])])
            return f"{ips[0]}-{ips[1]}-{ports[0]}-{ports[1]}-{row['proto']}"
        df['session_id'] = df.apply(get_directionless, axis=1)
        n_sessions = df['session_id'].nunique()
        print(f"Grouped into {n_sessions} directionless sessions.")
    else:
        print("Warning: Missing session key columns in UNSW-NB15 (expected srcip, sport, dstip, dsport).")
        df['session_id'] = np.arange(len(df))

    # Feature vector array
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col, 'srcip', 'sport', 'dstip', 'dsport']]
    
    print(f"UNSW-NB15 dimension d={len(feature_cols)}")
    
    # Save standard output
    df = df[['session_id', label_col] + feature_cols]
    out_path = os.path.join(out_dir, "unsw_nb15_processed.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    print(f"Class balance: {df[label_col].value_counts().to_dict()}")

def preprocess_cicids():
    print("\n--- Preprocessing CICIDS2017 ---")
    raw_dir = "data/raw/cicids2017"
    out_dir = "data/processed/cicids2017"
    os.makedirs(out_dir, exist_ok=True)
    
    files = glob.glob(f"{raw_dir}/**/*.csv", recursive=True)
    if not files:
        print(f"No CSVs found in {raw_dir}")
        return
        
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1", on_bad_lines='skip', low_memory=False)
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded total {len(df)} rows.")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    label_col = 'Label'
    if label_col not in df.columns:
        print("Error: Label column not found.")
        return
        
    # Convert string labels to binary (0=Benign, 1=Attack)
    df['Label_Binary'] = (df[label_col] != 'BENIGN').astype(int)
    
    # Handle Inf/NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).values.sum()
    nan_count = df[numeric_cols].isna().values.sum()
    print(f"Found {inf_count} Inf values and {nan_count} NaN values.")
    
    for c in numeric_cols:
        if np.isinf(df[c]).any():
            median_val = df.loc[~np.isinf(df[c]), c].median()
            df[c].replace([np.inf, -np.inf], median_val, inplace=True)
        if df[c].isna().any():
            median_val = df[c].median()
            df[c].fillna(median_val, inplace=True)
            
    # Group into directionless sessions
    if 'Flow ID' in df.columns:
        # Flow ID is usually src_ip-dst_ip-src_port-dst_port-protocol
        def get_directionless_flow(flow_id):
            parts = str(flow_id).split('-')
            if len(parts) >= 5:
                ips = sorted([parts[0], parts[1]])
                ports = sorted([parts[2], parts[3]])
                return f"{ips[0]}-{ips[1]}-{ports[0]}-{ports[1]}-{parts[4]}"
            return flow_id
        df['session_id'] = df['Flow ID'].apply(get_directionless_flow)
        
        orig_sessions = df['Flow ID'].nunique()
        new_sessions = df['session_id'].nunique()
        print(f"Directionless merging combined {orig_sessions - new_sessions} bidirectional flows.")
        print(f"Total unique sessions: {new_sessions}")
    else:
        print("Warning: 'Flow ID' not found.")
        df['session_id'] = np.arange(len(df))
        
    # Drop identifier columns
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Timestamp', label_col]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    feature_cols = [c for c in df.columns if c not in ['session_id', 'Label_Binary']]
    
    # Scale
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"CICIDS2017 dimension d={len(feature_cols)}")
    
    df = df[['session_id', 'Label_Binary'] + feature_cols]
    df.rename(columns={'Label_Binary': 'label'}, inplace=True)
    
    out_path = os.path.join(out_dir, "cicids2017_processed.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    print(f"Class balance: {df['label'].value_counts().to_dict()}")

if __name__ == '__main__':
    # preprocess_unsw() # already done successfully
    preprocess_cicids()
