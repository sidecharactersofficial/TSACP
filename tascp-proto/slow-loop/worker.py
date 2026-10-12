import sqlite3
import time
import requests
import json

DB_PATH = "../fast-loop/events.db"
ADMIN_URL = "http://localhost:3000/admin/params"

def run_loop():
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # Simple polling of latest sessions
            c.execute("SELECT DISTINCT session_id FROM events")
            sessions = c.fetchall()
            
            for (session_id,) in sessions:
                # Stub classification/signature logic
                # Normally, we analyze the trajectory here.
                
                # Push back updated params
                new_params = {
                    "gamma": 0.85,
                    "lambda": 0.15,
                    "k": 6.0,
                    "tau": 0.6
                }
                
                try:
                    res = requests.post(f"{ADMIN_URL}/{session_id}", json=new_params)
                    print(f"Updated {session_id}: {res.status_code}")
                except Exception as e:
                    print(f"Failed to update fast loop: {e}")
                    
            conn.close()
        except sqlite3.OperationalError:
            pass # DB might not exist yet
            
        time.sleep(5) # Lag-tolerant loop

if __name__ == "__main__":
    print("Starting slow loop worker...")
    run_loop()
