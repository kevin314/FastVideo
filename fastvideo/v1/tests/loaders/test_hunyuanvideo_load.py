import argparse
import glob
import json
import os
import pytest
import torch

from fastvideo.v1.distributed.parallel_state import (
    get_sequence_model_parallel_rank,
    get_sequence_model_parallel_world_size)
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.dits.hunyuanvideo import (
    HunyuanVideoTransformer3DModel as HunyuanVideoDit)
from fastvideo.v1.models.loader.fsdp_load import load_fsdp_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

LOCAL_RANK = 0
RANK = 0
WORLD_SIZE = 1

REFERENCE_LATENT = 0 ##TODO Find real latent


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuanvideo_distributed():
    logger.info(
        f"Initializing process: rank={RANK}, local_rank={LOCAL_RANK}, world_size={WORLD_SIZE}"
    )

    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")

    # Get tensor parallel info
    sp_rank = get_sequence_model_parallel_rank()
    sp_world_size = get_sequence_model_parallel_world_size()

    logger.info(
        f"Process rank {RANK} initialized with SP rank {sp_rank} in SP world size {sp_world_size}"
    )

    # load data/hunyuanvideo_community/transformer/config.json
    with open(
            "data/hunyuanvideo-community/HunyuanVideo/transformer/config.json") as f:
        config = json.load(f)
    # remove   "_class_name": "HunyuanVideoTransformer3DModel",   "_diffusers_version": "0.32.0.dev0",
    # TODO: write normalize config function
    config.pop("_class_name")
    config.pop("_diffusers_version")
    # load data/hunyuanvideo_community/transformer/*.safetensors
    weight_dir_list = glob.glob(
        "data/hunyuanvideo-community/HunyuanVideo/transformer/*.safetensors")
    # to str
    weight_dir_list = [str(path) for path in weight_dir_list]
    model = load_fsdp_model(HunyuanVideoDit,
                             init_params=config,
                             weight_dir_list=weight_dir_list,
                             device=torch.device(f"cuda:{LOCAL_RANK}"),
                             cpu_offload=False)

    model.eval()

    # Create random inputs for testing
    batch_size = 1
    seq_len = 3
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                16,
                                8,
                                16,
                                16,
                                device=device,
                                dtype=torch.bfloat16)
    chunk_per_rank = hidden_states.shape[2] // sp_world_size
    hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) *
                                  chunk_per_rank]

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len + 1,
                                        4096,
                                        device=device,
                                        dtype=torch.bfloat16)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)

    # Disable gradients for inference
    with torch.no_grad():
        # Run inference on model
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logger.info("Running inference on model")
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            logger.info("Model 1 inference completed")

    latent = output.double().sum().item()

    # Check if latents are similar
    diff_output_latents = abs(REFERENCE_LATENT - latent)
    logger.info(
        f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}"
    )
    assert diff_output_latents < 1e-4, f"Output latents differ significantly: max diff = {diff_output_latents}"


