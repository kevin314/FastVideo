#!/usr/bin/env python3
"""
RunPod Cleanup Script

This script finds and terminates RunPod instances created by GitHub Actions workflows.
"""

import json
import os
import sys
import requests

API_KEY = os.environ['RUNPOD_API_KEY']
RUN_ID = os.environ['GITHUB_RUN_ID']
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def get_job_ids():
    """Parse job IDs from environment variable, with error handling"""
    job_ids_str = os.environ.get('JOB_IDS', '["unit-test"]')
    try:
        job_ids = json.loads(job_ids_str)
        if not isinstance(job_ids, list):
            print(f"Warning: JOB_IDS is not a list. Using default.")
            return ["unit-test"]
        return job_ids
    except json.JSONDecodeError as e:
        print(f"Error parsing JOB_IDS: {e}. Using default.")
        return ["unit-test"]


def cleanup_pods():
    """Find and terminate RunPod instances"""
    job_ids = get_job_ids()

    print(f"RunPod Cleanup")
    print(f"Run ID: {RUN_ID}")
    print(f"Job IDs: {job_ids}")

    # Get all pods
    try:
        response = requests.get(PODS_API, headers=HEADERS)
        response.raise_for_status()
        pods = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pods: {e}")
        return False

    # Find and terminate pods created by this workflow run
    terminated_pods = []
    for pod in pods:
        pod_name = pod.get("name", "")
        pod_id = pod.get("id")

        # Check if this pod was created by one of our jobs
        if any(f"{job_id}-{RUN_ID}" in pod_name for job_id in job_ids):
            print(f"Found pod: {pod_id} ({pod_name})")
            try:
                print(f"Terminating pod {pod_id}...")
                term_response = requests.delete(f"{PODS_API}/{pod_id}",
                                                headers=HEADERS)
                term_response.raise_for_status()
                terminated_pods.append(pod_id)
                print(f"Successfully terminated pod {pod_id}")
            except requests.exceptions.RequestException as e:
                print(f"Error terminating pod {pod_id}: {e}")
    if terminated_pods:
        print(f"Terminated {len(terminated_pods)} pods: {terminated_pods}")
    else:
        print("No pods found to terminate.")

    return True


def main():
    """Main function"""
    success = cleanup_pods()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
