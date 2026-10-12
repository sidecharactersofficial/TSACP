import requests
import time

URL = "http://localhost:3000/predict"

def simulate_probe():
    session_id = "probe_session_1"
    
    # Fixed direction iterative probe to exercise rotation
    for i in range(20):
        features = [0.1 * i for _ in range(10)]
        res = requests.post(URL, json={
            "session_id": session_id,
            "features": features
        })
        print(f"Probe Response: {res.status_code} - {res.json()}")
        time.sleep(0.05)

if __name__ == "__main__":
    simulate_probe()
