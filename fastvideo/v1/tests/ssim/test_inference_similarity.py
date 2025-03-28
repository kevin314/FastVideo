import os
import pytest
import time
import re
import json
import imageio
import numpy as np
import torch
import torchvision
import subprocess
import sys
from einops import rearrange
from datetime import datetime
from pathlib import Path

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.tests.ssim.compute_ssim import compute_video_ssim_torchvision

from fastvideo.v1.entrypoints.cli.utils import launch_distributed

# Base parameters from the shell script
BASE_PARAMS = {
    "num_gpus": 2,
    "model_path": "data/FastHunyuan-diffusers",
    "height": 720,
    "width": 1280,
    "num_frames": 45,
    "num_inference_steps": 6,
    "guidance_scale": 1,
    "embedded_cfg_scale": 6,
    "flow_shift": 17,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 2,
    "vae_sp": True,
    "use_v1_transformer": True,
    "use_v1_vae": True,
    "use_v1_text_encoder": True,
    "fps": 24,
}

# Test prompts
TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
]


def write_ssim_results(output_dir, ssim_values, reference_path, generated_path, num_inference_steps, prompt):
    """
    Write SSIM results to a JSON file in the same directory as the generated videos.
    """
    try:
        print(f"Attempting to write SSIM results to directory: {output_dir}")

        # Check if directory exists and is writable
        if not os.path.exists(output_dir):
            print(f"Output directory does not exist, creating: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        if not os.access(output_dir, os.W_OK):
            print(f"WARNING: Output directory is not writable: {output_dir}")

        # Unpack SSIM values
        mean_ssim, min_ssim, max_ssim = ssim_values

        # Create a result object
        result = {
            "mean_ssim": mean_ssim,
            "min_ssim": min_ssim,
            "max_ssim": max_ssim,
            "reference_video": reference_path,
            "generated_video": generated_path,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "prompt": prompt
            }
        }

        # Write to a file named after the parameters
        test_name = f"steps{num_inference_steps}_{prompt[:100]}"
        result_file = os.path.join(output_dir, f"{test_name}_ssim.json")
        print(f"Writing JSON results to: {result_file}")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"SSIM results successfully written to {result_file}")
        return True
    except Exception as e:
        print(f"ERROR writing SSIM results: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Parameters to vary in the tests
@pytest.mark.parametrize("num_inference_steps", [5])
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_inference_similarity(num_inference_steps, prompt):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos')
    output_dir = os.path.join(base_output_dir, f'num_inference_steps={num_inference_steps}')
    output_video_name = f"{prompt[:100]}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    test_name = f"steps{num_inference_steps}_{prompt[:100]}"
 
    launch_args = [
        "--num-inference-steps", str(num_inference_steps),
        "--prompt", prompt,
        "--output-path", output_dir,
        "--model-path", BASE_PARAMS["model_path"],
        "--height", str(BASE_PARAMS["height"]),
        "--width", str(BASE_PARAMS["width"]),
        "--num-frames", str(BASE_PARAMS["num_frames"]),
        "--guidance-scale", str(BASE_PARAMS["guidance_scale"]),
        "--embedded-cfg-scale", str(BASE_PARAMS["embedded_cfg_scale"]),
        "--flow-shift", str(BASE_PARAMS["flow_shift"]),
        "--seed", str(BASE_PARAMS["seed"]),
        "--sp-size", str(BASE_PARAMS["sp_size"]),
        "--tp-size", str(BASE_PARAMS["tp_size"]),
        "--fps", str(BASE_PARAMS["fps"]),
    ]

    if BASE_PARAMS["use_v1_transformer"]:
        launch_args.append("--use-v1-transformer")
    if BASE_PARAMS["use_v1_vae"]:
        launch_args.append("--use-v1-vae")
    if BASE_PARAMS["use_v1_text_encoder"]:
        launch_args.append("--use-v1-text-encoder")
    if BASE_PARAMS["vae_sp"]:
        launch_args.append("--vae-sp")

    launch_distributed(
        num_gpus=BASE_PARAMS["num_gpus"], 
        args=launch_args
    )

    assert os.path.exists(output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = os.path.join(script_dir, 'reference_videos')

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100] in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        pytest.skip(f"No reference video found for prompt: {prompt}")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    print(f"Computing SSIM between {reference_video_path} and {output_dir}")
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, 
        generated_video_path, 
        use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]  # Extract mean SSIM for assertions
    print(f"SSIM computation complete, mean value: {mean_ssim}")
    print(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(
        output_dir,
        ssim_values,
        reference_video_path,
        generated_video_path,
        num_inference_steps,
        prompt
    )

    if not success:
        print("WARNING: Failed to write SSIM results to file")

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim}"

    print(f"SSIM for {test_name}: {mean_ssim}")

