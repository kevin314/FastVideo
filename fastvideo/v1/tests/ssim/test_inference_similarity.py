import os
import pytest
import subprocess
import time
from pathlib import Path

from fastvideo.v1.tests.ssim.compute_ssim import compute_video_ssim_torchvision

# Base parameters from the shell script
BASE_PARAMS = {
    "num_gpus": 4,
    "model_base": "data/FastHunyuan-diffusers",
    "height": 720,
    "width": 1280,
    "num_frames": 125,
    "num_inference_steps": 6,
    "guidance_scale": 1,
    "embedded_cfg_scale": 6,
    "flow_shift": 17,
    "seed": 1024,
    "sp_size": 4,
    "tp_size": 4,
    "vae_sp": True,
}

# Test prompts
TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
]

# Parameters to vary in the tests
@pytest.mark.parametrize("num_inference_steps", [4, 6])
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_inference_similarity(num_inference_steps, prompt):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output directory for generated videos
    output_dir = os.path.join(script_dir, 'generated_videos')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare parameters for this test run
    params = BASE_PARAMS.copy()
    params["num_inference_steps"] = num_inference_steps
    
    # Use default values for these parameters
    guidance_scale = BASE_PARAMS["guidance_scale"]
    embedded_cfg_scale = BASE_PARAMS["embedded_cfg_scale"]
    
    # Generate a unique name for this test configuration
    # Create a short identifier from the prompt (first few words)
    prompt_id = "_".join(prompt.split()[:3]).lower()
    prompt_id = ''.join(c if c.isalnum() or c == '_' else '' for c in prompt_id)
    
    test_name = f"test_steps{num_inference_steps}_{prompt_id}"
    output_video_name = f"{test_name}.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    
    # Run the inference command
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={params['num_gpus']}",
        "--master_port", str(29500 + int(time.time() % 1000)),  # Use a dynamic port to avoid conflicts
        "fastvideo/v1/sample/v1_fastvideo_inference.py",
        "--use-v1-transformer",
        "--use-v1-vae",
        "--use-v1-text-encoder",
        f"--sp_size", str(params["sp_size"]),
        f"--tp_size", str(params["tp_size"]),
        f"--height", str(params["height"]),
        f"--width", str(params["width"]),
        f"--num_frames", str(params["num_frames"]),
        f"--num_inference_steps", str(params["num_inference_steps"]),
        f"--guidance_scale", str(guidance_scale),
        f"--embedded_cfg_scale", str(embedded_cfg_scale),
        f"--flow_shift", str(params["flow_shift"]),
        f"--prompt", prompt,
        f"--seed", str(params["seed"]),
        f"--output_path", output_dir,
        f"--model_path", params["model_base"],
    ]
    
    if params["vae_sp"]:
        cmd.append("--vae-sp")
        
    # Add a parameter to name the output file according to our test
    cmd.extend(["--output_name", output_video_name])
    
    print(f"Running inference with command: {' '.join(cmd)}")
    
    # Run the inference process
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Inference process failed with error: {e}")
    
    # Check if the output video was generated
    assert os.path.exists(output_video_path), f"Output video was not generated at {output_video_path}"
    
    # Define path to the reference video
    reference_folder = os.path.join(script_dir, 'reference_videos')
    reference_video_path = os.path.join(reference_folder, output_video_name)
    
    # Check if reference video exists
    if not os.path.exists(reference_video_path):
        # If no reference exists, we can optionally save this as a reference
        # for future tests, but for now we'll just skip the comparison
        pytest.skip(f"No reference video found at {reference_video_path}")
    
    # Compute SSIM between generated and reference videos
    ssim_value = compute_video_ssim_torchvision(
        reference_video_path, 
        output_video_path, 
        use_ms_ssim=True
    )
    
    # Assert that SSIM is above a threshold
    min_acceptable_ssim = 0.85  # This threshold can be adjusted based on your requirements
    assert ssim_value >= min_acceptable_ssim, f"SSIM value {ssim_value} is below threshold {min_acceptable_ssim}"
    
    # Print the SSIM value for reference
    print(f"SSIM for {test_name}: {ssim_value}")
