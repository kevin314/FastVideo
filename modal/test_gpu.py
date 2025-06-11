import modal

app = modal.App()
# image = modal.Image.debian_slim().pip_install("torch")

image = (
    modal.Image.from_registry("ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest", add_python="3.12")
    .apt_install("cmake", "pkg-config", "build-essential", "curl", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .pip_install("torch==2.6.0")
    .run_commands("/bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate fastvideo-dev && cd /FastVideo && pip install -e .[dev]'")
)

@app.function(gpu="L40S:2", image=image, timeout=600)
def run():
    # import torch
    import subprocess
    import sys
    # print(torch.cuda.is_available())
    
    # Activate conda environment and run
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    echo 'Checking fastvideo installation:' && 
    pip show fastvideo && 
    echo 'Testing torch:' && 
    python -c 'import torch; print("Torch CUDA:", torch.cuda.is_available())' && 
    echo 'Running fastvideo:' && 
    fastvideo
    """
        # fastvideo generate --model-path FastVideo/FastHunyuan-diffusers --num-gpus 2 --prompt 'a dog on a beach' --num-inference-steps 6 --num-frames 45
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=True)
    
    return result