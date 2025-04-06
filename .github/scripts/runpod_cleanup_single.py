import os
import sys
import requests
import uuid

API_KEY = os.environ['RUNPOD_API_KEY']
RUN_ID = os.environ.get('GITHUB_RUN_ID', str(uuid.uuid4()))
JOB_ID = os.environ.get('JOB_ID')
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def cleanup_single_pod():
    """Find and terminate a single RunPod instance for a specific job"""
    if not JOB_ID:
        print("Error: JOB_ID environment variable is required")
        sys.exit(1)

    print(f"RunPod Single Pod Cleanup")
    print(f"Run ID: {RUN_ID}")
    print(f"Job ID: {JOB_ID}")

    # Get all pods associated with RunPod API_KEY
    try:
        response = requests.get(PODS_API, headers=HEADERS)
        response.raise_for_status()
        pods = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pods: {e}")
        sys.exit(1)

    # Find and terminate the pod created by this specific job
    pod_name_pattern = f"{JOB_ID}-{RUN_ID}"
    terminated_pod = None
    
    for pod in pods:
        pod_name = pod.get("name", "")
        pod_id = pod.get("id")

        if pod_name_pattern in pod_name:
            print(f"Found pod: {pod_id} ({pod_name})")
            try:
                print(f"Terminating pod {pod_id}...")
                term_response = requests.delete(f"{PODS_API}/{pod_id}",
                                              headers=HEADERS)
                term_response.raise_for_status()
                terminated_pod = pod_id
                print(f"Successfully terminated pod {pod_id}")
                break  # Exit after finding and terminating the first matching pod
            except requests.exceptions.RequestException as e:
                print(f"Error terminating pod {pod_id}: {e}")
                sys.exit(1)
    
    if terminated_pod:
        print(f"Terminated pod: {terminated_pod}")
    else:
        print(f"No pod found matching pattern: {pod_name_pattern}")


def main():
    cleanup_single_pod()


if __name__ == "__main__":
    main()
