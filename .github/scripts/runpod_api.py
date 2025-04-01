import os
import json
import time
import requests
import sys
import argparse
import subprocess


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run tests on RunPod GPU')
    parser.add_argument('--gpu-type', type=str, help='GPU type to use')
    parser.add_argument('--gpu-count', type=int, help='Number of GPUs to use', default=1)
    parser.add_argument('--test-command', type=str, help='Test command to run')
    parser.add_argument('--disk-size',
                        type=int,
                        default=20,
                        help='Container disk size in GB (default: 20)')
    parser.add_argument('--volume-size',
                        type=int,
                        default=20,
                        help='Persistent volume size in GB (default: 20)')
    parser.add_argument(
        '--image',
        type=str,
        default='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
        help='Docker image to use')
    return parser.parse_args()


# Configuration
args = parse_arguments()
API_KEY = os.environ['RUNPOD_API_KEY']
GITHUB_SHA = os.environ['GITHUB_SHA']
GITHUB_REF = os.environ.get('GITHUB_REF', 'unknown')
GITHUB_REPOSITORY = os.environ['GITHUB_REPOSITORY']
RUN_ID = os.environ['GITHUB_RUN_ID']

# API endpoints
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def create_pod():
    """Create a RunPod instance"""
    print(f"Creating RunPod instance with GPU: {args.gpu_type}...")
    payload = {
        "name": f"github-test-{RUN_ID}",
        "containerDiskInGb": args.disk_size,
        "volumeInGb": args.volume_size,
        "env": {
            "GITHUB_SHA": GITHUB_SHA,
            "GITHUB_REF": GITHUB_REF
        },
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "imageName": args.image
    }

    response = requests.post(PODS_API, headers=HEADERS, json=payload)
    response_data = response.json()
    print(f"Response: {json.dumps(response_data, indent=2)}")

    return response_data["id"]


def wait_for_pod(pod_id):
    """Wait for pod to be in RUNNING state and fully ready with SSH access"""
    print("Waiting for RunPod to be ready...")
    
    # First wait for RUNNING status
    while True:
        response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
        pod_data = response.json()
        status = pod_data["desiredStatus"]
        
        if status == "RUNNING":
            print("RunPod is running! Now waiting for ports to be assigned...")
            break
            
        print(f"Current status: {status}, waiting...")
        time.sleep(2)
    
    # Then wait for SSH and public IP
    while True:
        response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
        pod_data = response.json()
        port_mappings = pod_data.get("portMappings")
        
        if (port_mappings is not None and 
            "22" in port_mappings and 
            pod_data.get("publicIp", "") != ""):
            print("RunPod is ready with SSH access!")
            print(f"SSH IP: {pod_data['publicIp']}")
            print(f"SSH Port: {port_mappings['22']}")
            break
            
        print("Waiting for SSH port and public IP to be available...")
        time.sleep(10)


def execute_command(pod_id):
    """Execute command on the pod via SSH using system SSH client"""
    print(f"Running command: {args.test_command}")
    
    # Get pod information for SSH connection
    response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
    pod_data = response.json()
    ssh_ip = pod_data["publicIp"]
    ssh_port = pod_data["portMappings"]["22"]
    
    # Prepare commands with Conda setup before git clone
    setup_steps = [
        "cd /workspace",
        # Set up Conda first
        "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
        "bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3",
        "source $HOME/miniconda3/bin/activate",
        "conda create --name venv python=3.10.0 -y",
        "conda activate venv",
        # Now clone the repository
        f"git clone https://github.com/{GITHUB_REPOSITORY}.git",
        f"cd $(basename {GITHUB_REPOSITORY})",
        f"git checkout {GITHUB_SHA}",
        # Run the test command in the Conda environment
        args.test_command
    ]
    remote_command = " && ".join(setup_steps)
    
    # Build SSH command
    ssh_command = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",  # Don't ask to confirm host key
        "-o", "UserKnownHostsFile=/dev/null",  # Don't save host key
        "-p", str(ssh_port),
        f"root@{ssh_ip}",
        remote_command
    ]
    
    print(f"Connecting to {ssh_ip}:{ssh_port}...")
    
    try:
        # Execute SSH command with real-time output
        process = subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Capture output in strings for return value
        stdout_lines = []
        
        # Print output in real-time
        print("Command output:")
        
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            stdout_lines.append(line)
            
        # Wait for process to complete
        process.wait()
        
        # Get return code
        return_code = process.returncode
        success = return_code == 0
        
        # Combine output lines
        stdout_str = "".join(stdout_lines)
        
        if success:
            print("Command executed successfully")
        else:
            print(f"Command failed with exit code {return_code}")
        
        # Return results
        result = {
            "success": success,
            "stdout": stdout_str,
            "stderr": ""  # Empty since stderr was redirected to stdout
        }
        return result
        
    except Exception as e:
        print(f"Error executing SSH command: {str(e)}")
        result = {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }
        return result


def terminate_pod(pod_id):
    """Terminate the pod"""
    print("Terminating RunPod...")
    requests.delete(f"{PODS_API}/{pod_id}", headers=HEADERS)
    print("RunPod terminated")


def main():
    pod_id = None
    try:
        pod_id = create_pod()
        wait_for_pod(pod_id)
        result = execute_command(pod_id)

        if result.get("error") is not None:
            print("Command failed!")
            sys.exit(1)
    finally:
        if pod_id:
            terminate_pod(pod_id)


if __name__ == "__main__":
    main()
