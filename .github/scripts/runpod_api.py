import argparse
import json
import os
import subprocess
import sys
import time

import requests


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests on RunPod GPU")
    parser.add_argument("--gpu-type", type=str, help="GPU type to use")
    parser.add_argument(
        "--gpu-count", type=int, help="Number of GPUs to use", default=1
    )
    parser.add_argument("--test-command", type=str, help="Test command to run")
    parser.add_argument(
        "--disk-size",
        type=int,
        default=20,
        help="Container disk size in GB (default: 20)",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=20,
        help="Persistent volume size in GB (default: 20)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        help="Docker image to use",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["create", "execute", "delete"],
        default="execute",
        help="Action to perform: create pod, execute command, or delete pod",
    )
    parser.add_argument(
        "--pod-id", type=str, help="Pod ID for execute or delete actions"
    )
    return parser.parse_args()


args = parse_arguments()
API_KEY = os.environ["RUNPOD_API_KEY"]
RUN_ID = os.environ.get("GITHUB_RUN_ID", "")
JOB_ID = os.environ.get("JOB_ID", "")
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


def create_pod():
    """Create a RunPod instance"""
    if not RUN_ID or not JOB_ID:
        raise ValueError("No RUN_ID or JOB_ID provided for pod creation")
    print(f"Creating RunPod instance with GPU: {args.gpu_type}...")
    payload = {
        "name": f"fastvideo-{JOB_ID}-{RUN_ID}",
        "containerDiskInGb": args.disk_size,
        "volumeInGb": args.volume_size,
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "imageName": args.image,
    }

    response = requests.post(PODS_API, headers=HEADERS, json=payload)
    response_data = response.json()
    print(f"Response: {json.dumps(response_data, indent=2)}")

    pod_id = response_data["id"]
    print(f"Created pod with ID: {pod_id}")

    # Set GitHub Actions output
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"pod-id={pod_id}\n")

    return pod_id


def get_pod_info(pod_id):
    """Get pod information"""
    response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
    return response.json()


def wait_for_pod(pod_id):
    """Wait for pod to be in RUNNING state and fully ready with SSH access"""
    print("Waiting for RunPod to be ready...")

    # First wait for RUNNING status
    max_attempts = 10
    attempts = 0
    while attempts < max_attempts:
        pod_data = get_pod_info(pod_id)
        status = pod_data["desiredStatus"]

        if status == "RUNNING":
            print("RunPod is running! Now waiting for ports to be assigned...")
            break

        print(
            f"Current status: {status}, waiting... (attempt {attempts + 1}/{max_attempts})"
        )
        time.sleep(2)
        attempts += 1

    if attempts >= max_attempts:
        raise TimeoutError(
            "Timed out waiting for RunPod to reach RUNNING state"
        )

    # Wait for ports to be assigned
    max_attempts = 6
    attempts = 0
    while attempts < max_attempts:
        pod_data = get_pod_info(pod_id)
        port_mappings = pod_data.get("portMappings")

        if (
            port_mappings is not None
            and "22" in port_mappings
            and pod_data.get("publicIp", "") != ""
        ):
            print("RunPod is ready with SSH access!")
            print(f"SSH IP: {pod_data['publicIp']}")
            print(f"SSH Port: {port_mappings['22']}")
            break

        print(
            f"Waiting for SSH port and public IP to be available... (attempt {attempts + 1}/{max_attempts})"
        )
        time.sleep(10)
        attempts += 1

    if attempts >= max_attempts:
        raise TimeoutError("Timed out waiting for RunPod SSH access")


def setup_repository(pod_id):
    """Copy the repository to the pod and set up the environment"""
    print("Setting up repository on RunPod...")

    pod_data = get_pod_info(pod_id)
    ssh_ip = pod_data["publicIp"]
    ssh_port = pod_data["portMappings"]["22"]

    # Copy the repository to the pod using scp
    repo_dir = os.path.abspath(os.getcwd())
    repo_name = os.path.basename(repo_dir)

    print(f"Copying repository from {repo_dir} to RunPod...")

    try:
        # Create tarball of repository
        tar_command = [
            "tar",
            "-czf",
            "/tmp/repo.tar.gz",
            "-C",
            os.path.dirname(repo_dir),
            repo_name,
        ]
        subprocess.run(tar_command, check=True)

        # Copy the tarball to the pod
        scp_command = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=10",
            "-P",
            str(ssh_port),
            "/tmp/repo.tar.gz",
            f"root@{ssh_ip}:/tmp/",
        ]
        subprocess.run(scp_command, check=True)

        # Set up the environment
        setup_steps = [
            "cd /workspace",
            "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            "bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3",
            "source $HOME/miniconda3/bin/activate",
            "conda create --name venv python=3.10.0 -y",
            "conda activate venv",
            "mkdir -p /workspace",
            "tar -xzf /tmp/repo.tar.gz --no-same-owner -C /workspace/",
            f"cd /workspace/{repo_name}"
        ]
        remote_command = " && ".join(setup_steps)

        ssh_command = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=10",
            "-p",
            str(ssh_port),
            f"root@{ssh_ip}",
            remote_command,
        ]

        print(f"Setting up environment on {ssh_ip}:{ssh_port}...")

        process = subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        stdout_lines = []

        print("Setup output:")

        for line in iter(process.stdout.readline, ""):
            print(line.strip())
            stdout_lines.append(line)

        process.wait()

        return_code = process.returncode
        if return_code != 0:
            stdout_str = "".join(stdout_lines)
            raise RuntimeError(
                f"Repository setup failed with exit code {return_code}. Output: {stdout_str}"
            )

        print("Repository setup complete!")
        return True

    except Exception as e:
        print(f"Error setting up repository: {str(e)}")
        raise


def execute_command(pod_id):
    """Execute command on the pod via SSH using system SSH client"""
    if not args.test_command:
        print("No test command provided, skipping execution")
        return {"success": True}
        
    print(f"Running command: {args.test_command}")

    pod_data = get_pod_info(pod_id)
    ssh_ip = pod_data["publicIp"]
    ssh_port = pod_data["portMappings"]["22"]
    
    # Get repository name for the working directory
    repo_dir = os.path.abspath(os.getcwd())
    repo_name = os.path.basename(repo_dir)
    
    # Prepend directory change and conda environment activation
    full_command = (
        f"cd /workspace/{repo_name} && "
        f"export PATH=$HOME/miniconda3/bin:$PATH && "
        f"source $HOME/miniconda3/bin/activate && "
        f"conda activate venv && "
        f"{args.test_command}"
    )

    ssh_command = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=10",
        "-p",
        str(ssh_port),
        f"root@{ssh_ip}",
        full_command,
    ]

    print(f"Connecting to {ssh_ip}:{ssh_port}...")
    print(f"Working directory: /workspace/{repo_name}")
    print(f"Using conda environment: venv")

    try:
        process = subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        stdout_lines = []

        print("Command output:")

        for line in iter(process.stdout.readline, ""):
            print(line.strip())
            stdout_lines.append(line)

        process.wait()

        return_code = process.returncode
        success = return_code == 0

        stdout_str = "".join(stdout_lines)

        if success:
            print("Command executed successfully")
        else:
            print(f"Command failed with exit code {return_code}")

        result = {
            "success": success,
            "return_code": return_code,
            "stdout": stdout_str,
            "stderr": "",
        }
        return result

    except Exception as e:
        print(f"Error executing SSH command: {str(e)}")
        result = {"success": False, "error": str(e), "stdout": "", "stderr": ""}
        return result


def delete_pod(pod_id):
    """Delete the pod"""
    print(f"Deleting RunPod {pod_id}...")
    requests.delete(f"{PODS_API}/{pod_id}", headers=HEADERS)
    print(f"Deleted pod {pod_id}")


def main():
    if args.action == "create":
        try:
            pod_id = create_pod()
            wait_for_pod(pod_id)
            setup_repository(pod_id)  # Always set up repository on create
            print(f"POD_ID={pod_id}")
        except Exception as e:
            print(f"Error creating pod: {str(e)}")
            sys.exit(1)

    elif args.action == "execute":
        try:
            if not args.pod_id:
                raise ValueError("No pod ID provided for execute action")

            result = execute_command(args.pod_id)

            if result.get("error") is not None:
                print(f"Error executing command: {result['error']}")
                sys.exit(1)

            if not result.get("success", False):
                print("Command failed - check the output above for details")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    elif args.action == "delete":
        try:
            if not args.pod_id:
                raise ValueError("No pod ID provided for delete action")

            delete_pod(args.pod_id)
        except Exception as e:
            print(f"Error deleting pod: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
