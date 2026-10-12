import requests
import random
import time

URL = "http://localhost:3000/predict"

def simulate_benign():
    session_id = f"benign_{random.randint(1000, 9999)}"
    
    for _ in range(20):
        features = [random.random() for _ in range(10)]
        res = requests.post(URL, json={
            "session_id": session_id,
            "features": features
        })
        print(f"Benign Response: {res.status_code}")
        time.sleep(0.1)

if __name__ == "__main__":
    simulate_benign()
