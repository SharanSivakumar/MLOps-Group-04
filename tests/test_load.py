import subprocess
import pandas as pd
import pytest
import os

def test_ecg_performance():
    # Define your parameters
    locust_file = "tests/performancetests/locustfile.py"
    host = "https://ecg-api-579499894470.europe-north1.run.app"
    users = 5
    spawn_rate = 1
    run_time = "1m"
    csv_prefix = "performance_results"

    command = [
        "locust",
        "-f", locust_file,
        "--headless",
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", run_time,
        "--host", host,
        "--csv", csv_prefix,
        "--only-summary"
    ]


    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Locust failed: {result.stderr}"

    df = pd.read_csv(f"{csv_prefix}_stats.csv")
    agg_stats = df[df["Name"] == "Aggregated"]
    p95_latency = agg_stats["95%"].values[0]
    
    assert p95_latency < 1200, f"Latency too high: {p95_latency}ms"

    for suffix in ["_stats.csv", "_stats_history.csv", "_failures.csv", "_exceptions.csv"]:
        if os.path.exists(csv_prefix + suffix):
            os.remove(csv_prefix + suffix)