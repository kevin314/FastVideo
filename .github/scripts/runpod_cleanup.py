#!/usr/bin/env python3
"""
RunPod Cleanup Script

This script finds and terminates RunPod instances created by GitHub Actions workflows.
"""

import os
import sys
import requests

API_KEY = os.environ['RUNPOD_API_KEY']
RUN_ID = os.environ['GITHUB_RUN_ID']
JOB_ID = os.environ['JOB_ID']
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def cleanup_pods():
    """Find and terminate RunPod instances"""
    print(f"RunPod Cleanup")
    print(f"Run ID: {RUN_ID}")
    print(f"Job ID: {JOB_ID}")

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

        # Check if this pod was created by this specific job
        if f"{JOB_ID}-{RUN_ID}" in pod_name:
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
