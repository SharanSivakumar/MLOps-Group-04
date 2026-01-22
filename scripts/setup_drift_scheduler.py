"""Setup Cloud Scheduler to run drift detection periodically."""

import subprocess
import sys


def setup_drift_monitoring(project_id: str, region: str = "europe-north1"):
    """
    Setup Cloud Scheduler to periodically check for drift.

    Run this script after deploying the drift detection API:
    uv run python scripts/setup_drift_scheduler.py <project-id>
    """

    # Get the drift API URL
    try:
        result = subprocess.run(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                "drift-detection-api",
                "--region",
                region,
                "--format=value(status.url)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        drift_api_url = result.stdout.strip()
        print(f"Drift API URL: {drift_api_url}")
    except subprocess.CalledProcessError as e:
        print(f"Error getting drift API URL: {e}")
        sys.exit(1)

    # Create Cloud Scheduler job to check drift daily
    scheduler_commands = [
        # Delete existing job if it exists
        [
            "gcloud",
            "scheduler",
            "jobs",
            "delete",
            "drift-check-daily",
            "--location",
            region,
            "--quiet",
        ],
        # Create new job
        [
            "gcloud",
            "scheduler",
            "jobs",
            "create",
            "http",
            "drift-check-daily",
            "--location",
            region,
            "--schedule",
            "0 2 * * *",  # Daily at 2 AM UTC
            "--uri",
            f"{drift_api_url}/check_drift",
            "--http-method",
            "POST",
            "--oidc-service-account-email",
            f"{project_id}@appspot.gserviceaccount.com",
            "--time-zone",
            "UTC",
            "--message-body",
            '{"bucket_name": "psychic-iridium-484208-c3-mlops-data"}',
            "--headers",
            "Content-Type=application/json",
        ],
    ]

    # Delete existing job (may fail if doesn't exist)
    try:
        subprocess.run(scheduler_commands[0], check=False)
    except Exception:
        pass

    # Create new job
    try:
        subprocess.run(scheduler_commands[1], check=True)
        print("✓ Cloud Scheduler job 'drift-check-daily' created successfully")
        print("  Schedule: Daily at 2 AM UTC")
        print(f"  Target: {drift_api_url}/check_drift")
    except subprocess.CalledProcessError as e:
        print(f"Error creating scheduler job: {e}")
        sys.exit(1)

    # Create alert policy for drift detection
    print("\nNext steps:")
    print("1. Go to Cloud Monitoring: https://console.cloud.google.com/monitoring")
    print("2. Create an alerting policy:")
    print("   - Condition: Cloud Run service 'drift-detection-api' returns HTTP 200")
    print("   - Notification: Email/Slack when drift is detected")
    print("3. Check logs to monitor drift detection runs")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/setup_drift_scheduler.py <project-id>")
        sys.exit(1)

    project_id = sys.argv[1]
    setup_drift_monitoring(project_id)
