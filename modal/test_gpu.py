import modal

app = modal.App()
# image = modal.Image.debian_slim().pip_install("torch")

# Base image for all tests
image = (
    modal.Image.from_registry("ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest", add_python="3.12")
    .apt_install("cmake", "pkg-config", "build-essential", "curl", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .pip_install("torch==2.6.0", "pytest")
    .run_commands("/bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate fastvideo-dev && cd /FastVideo && pip install -e .[test]'")
)

@app.function(gpu="L40S:1", image=image, timeout=1800, mounts=[modal.Mount.from_local_dir(".", remote_path="/FastVideo")])
def run_encoder_tests():
    """Run encoder tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    pytest ./fastvideo/v1/tests/encoders -s
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    return result.returncode

@app.function(gpu="L40S:1", image=image, timeout=1800, mounts=[modal.Mount.from_local_dir(".", remote_path="/FastVideo")])
def run_vae_tests():
    """Run VAE tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    pytest ./fastvideo/v1/tests/vaes -s
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    return result.returncode

@app.function(gpu="L40S:1", image=image, timeout=1800, mounts=[modal.Mount.from_local_dir(".", remote_path="/FastVideo")])
def run_transformer_tests():
    """Run transformer tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    pytest ./fastvideo/v1/tests/transformers -s
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    return result.returncode

@app.function(gpu="L40S:2", image=image, timeout=3600, mounts=[modal.Mount.from_local_dir(".", remote_path="/FastVideo")])
def run_ssim_tests():
    """Run SSIM tests on 2x L40S GPUs"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    pytest ./fastvideo/v1/tests/ssim -vs
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    return result.returncode

# Keep the original function for basic testing
@app.function(gpu="L40S:2", image=image, timeout=600, mounts=[modal.Mount.from_local_dir(".", remote_path="/FastVideo")])
def run():
    """Basic test function"""
    import subprocess
    import sys
    
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    echo 'Checking fastvideo installation:' && 
    pip show fastvideo && 
    echo 'Testing torch:' && 
    python -c 'import torch; print("Torch CUDA:", torch.cuda.is_available())' && 
    echo 'Running fastvideo:' && 
    
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=True)
    
    return result