#!/usr/bin/env python3
import os
import subprocess
import sys

def main():


    out_dir = "data/raw/unsw_nb15"
    os.makedirs(out_dir, exist_ok=True)
    
    # Download from Kaggle
    print(f"Downloading UNSW-NB15 from mrwellsdavid/unsw-nb15 to {out_dir}...")
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download", "-d", "mrwellsdavid/unsw-nb15",
            "-p", out_dir, "--unzip"
        ])
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)
        
    print("Download complete. Running row-count sanity checks...")
    train_path = os.path.join(out_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(out_dir, "UNSW_NB15_testing-set.csv")
    
    if os.path.exists(train_path):
        train_rows = sum(1 for _ in open(train_path, "r", encoding="utf-8"))
        print(f"Found training set with {train_rows} rows.")
    if os.path.exists(test_path):
        test_rows = sum(1 for _ in open(test_path, "r", encoding="utf-8"))
        print(f"Found testing set with {test_rows} rows.")

if __name__ == '__main__':
    main()
