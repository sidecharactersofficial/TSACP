#!/usr/bin/env python3
import os
import subprocess
import sys

def main():


    out_dir = "data/raw/cicids2017"
    os.makedirs(out_dir, exist_ok=True)
    
    # Download from Kaggle
    dataset_id = "pshikk/cicids2017-untampered"
    
    print(f"Downloading CICIDS2017 from Kaggle ({dataset_id})...")
    
    # Download the dataset as a zip
    zip_path = os.path.join(out_dir, "cicids2017-untampered.zip")
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download", "-d", dataset_id,
            "-p", out_dir
        ])
        
        # Unzip specifically the GeneratedLabelledFlows/TrafficLabelling CSVs
        print("Unzipping GeneratedLabelledFlows/TrafficLabelling CSVs...")
        subprocess.run([
            "unzip", "-q", "-j", zip_path, 
            "GeneratedLabelledFlows/TrafficLabelling /*.csv", 
            "-d", out_dir
        ], check=False)

        # Clean up
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)
        
    print("Download complete. Checking downloaded files...")
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith('.csv'):
                path = os.path.join(root, f)
                rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore"))
                print(f"File {f}: {rows} rows.")

if __name__ == '__main__':
    main()
