import os
import sys
import time
import numpy as np
import asyncio
import aiohttp
import subprocess

async def fetch(session, url, data):
    start = time.perf_counter()
    async with session.post(url, json=data) as response:
        await response.read()
        end = time.perf_counter()
        return (end - start) * 1000  # convert to ms

async def benchmark(url, num_requests, concurrency):
    print(f"Benchmarking {url} with {num_requests} requests (concurrency={concurrency})...")
    
    # Pre-generate request payloads
    payloads = []
    for i in range(num_requests):
        # We will cycle through 10 different session IDs to simulate concurrent sessions
        session_id = f"sess_{i % 10}"
        features = np.random.randn(10).tolist()
        payloads.append({"session_id": session_id, "features": features})
        
    latencies = []
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for payload in payloads:
            tasks.append(asyncio.create_task(fetch(session, url, payload)))
            
            # Throttle task submission slightly to maintain concurrency limit without memory explosion
            if len(tasks) >= concurrency:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                latencies.extend([t.result() for t in done])
                tasks = list(pending)
                
        if tasks:
            done, _ = await asyncio.wait(tasks)
            latencies.extend([t.result() for t in done])
            
    return np.array(latencies)

def run_exp7():
    print("Starting fast-loop server...")
    proto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tascp-proto", "fast-loop"))
    server_process = subprocess.Popen(["cargo", "run", "--release"], cwd=proto_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        # Give it a second to start
        time.sleep(2)
        
        # Warmup
        url = "http://127.0.0.1:3000/predict"
        print("Running warmup...")
        asyncio.run(benchmark(url, 1000, 10))
        
        # Actual benchmark
        num_requests = 10000
        concurrency = 100
        latencies = asyncio.run(benchmark(url, num_requests, concurrency))
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean_lat = np.mean(latencies)
        
        print("\n=== EXP 7: Latency Benchmark SUMMARY ===")
        print(f"Total Requests: {num_requests}")
        print(f"Concurrency:    {concurrency}")
        print(f"Mean Latency:   {mean_lat:.3f} ms")
        print(f"p50 Latency:    {p50:.3f} ms")
        print(f"p95 Latency:    {p95:.3f} ms")
        print(f"p99 Latency:    {p99:.3f} ms")
        
        summary_path = os.path.join(os.path.dirname(__file__), "exp7_summary.md")
        with open(summary_path, "w") as f:
            f.write("# Experiment 7 Summary: Latency Benchmark\n\n")
            f.write("This experiment evaluates the latency of the TASCP fast-loop architecture under concurrent load to verify the sub-millisecond per-request target.\n\n")
            f.write("### Results Summary\n\n")
            f.write("| Metric | Value (ms) |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| **Total Requests** | {num_requests} |\n")
            f.write(f"| **Concurrency** | {concurrency} |\n")
            f.write(f"| **Mean Latency** | {mean_lat:.3f} ms |\n")
            f.write(f"| **p50 Latency** | {p50:.3f} ms |\n")
            f.write(f"| **p95 Latency** | {p95:.3f} ms |\n")
            f.write(f"| **p99 Latency** | {p99:.3f} ms |\n\n")
            f.write("### Key Findings\n")
            if p95 < 1.0:
                f.write("1. **Sub-millisecond target achieved**: The fast-loop successfully tracks session momentum, calculates $\epsilon_p$, applies the Givens rotation, and pushes telemetry to the event store in under 1 ms for the vast majority of requests.\n")
            else:
                f.write("1. **Sub-millisecond target missed**: The prototype implementation failed to consistently serve requests in under 1 ms under the tested concurrent load.\n")
            
    finally:
        print("Shutting down server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    run_exp7()
