from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch  # type: ignore

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video, pred_noise_to_x_bound
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

try:
    from fastvideo.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False
    SlidingTileAttentionBackend = None  # type: ignore

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False
    VideoSparseAttentionBackend = None  # type: ignore

logger = init_logger(__name__)


@dataclass
class BlockProcessingContext:
    """Dataclass contains for block processing."""

    batch: ForwardBatch

    block_idx: int
    start_index: int

    kv_cache1: list[dict[Any, Any]]
    kv_cache2: list[dict[Any, Any]] | None
    kv_cache_mouse: list[dict[Any, Any]] | None
    kv_cache_keyboard: list[dict[Any, Any]] | None
    crossattn_cache: list[dict[Any, Any]]

    timesteps: torch.Tensor
    block_sizes: list[int]
    noise_pool: list[torch.Tensor] | None

    fastvideo_args: FastVideoArgs
    target_dtype: torch.dtype
    autocast_enabled: bool
    boundary_timestep: float | None
    high_noise_timesteps: torch.Tensor | None
    context_noise: float

    use_scheduler_step: bool

    image_kwargs: dict[str, Any]
    pos_cond_kwargs: dict[str, Any]

    def get_kv_cache(self, timestep_val: float) -> list[dict[Any, Any]]:
        if self.boundary_timestep is not None:
            if timestep_val >= self.boundary_timestep:
                return self.kv_cache1
            else:
                assert self.kv_cache2 is not None, "kv_cache2 is not initialized"
                return self.kv_cache2
        return self.kv_cache1


class MatrixGameCausalDenoisingStage(DenoisingStage):

    def __init__(self,
                 transformer,
                 scheduler,
                 pipeline=None,
                 transformer_2=None,
                 vae=None) -> None:
        super().__init__(transformer, scheduler, pipeline, transformer_2, vae)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.vae = vae
        self.num_transformer_blocks = len(self.transformer.blocks)

        if hasattr(self.transformer, 'config') and hasattr(
                self.transformer.config, 'arch_config'):
            self.num_frame_per_block = getattr(
                self.transformer.config.arch_config, 'num_frames_per_block',
                getattr(self.transformer, 'num_frame_per_block', 1))
            self.sliding_window_num_frames = getattr(
                self.transformer.config.arch_config,
                'sliding_window_num_frames', 15)
        else:
            self.num_frame_per_block = getattr(self.transformer,
                                               'num_frame_per_block', 1)
            self.sliding_window_num_frames = 15

        try:
            self.local_attn_size = getattr(self.transformer, "local_attn_size",
                                           -1)
        except Exception:
            self.local_attn_size = -1

        assert self.num_frame_per_block > 0, (
            f"num_frame_per_block must be positive, got {self.num_frame_per_block}"
        )

        logger.info(
            "MatrixGame causal inference initialized: "
            "local_attn_size=%s, num_frame_per_block=%s, "
            "num_transformer_blocks=%s, num_attention_heads=%s, "
            "attention_head_dim=%s, hidden_size=%s, sliding_window=%s",
            self.local_attn_size, self.num_frame_per_block,
            self.num_transformer_blocks, self.transformer.num_attention_heads,
            getattr(self.transformer, 'attention_head_dim', 'N/A'),
            getattr(self.transformer, 'hidden_size', 'N/A'),
            self.sliding_window_num_frames)

        self.action_config = getattr(self.transformer, 'action_config', {})
        self.use_action_module = len(self.action_config) > 0

        self._streaming_initialized: bool = False
        self._streaming_ctx: BlockProcessingContext | None = None

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_size = self.transformer.patch_size
        patch_ratio = patch_size[-1] * patch_size[-2]
        self.frame_seq_length = latent_seq_length // patch_ratio

        dmd_denoising_steps = getattr(fastvideo_args.pipeline_config,
                                       'dmd_denoising_steps', None)
        if dmd_denoising_steps is None:
            self.scheduler.set_timesteps(batch.num_inference_steps,
                                         device=get_local_torch_device())
            timesteps = self.scheduler.timesteps
        else:
            timesteps = torch.tensor(dmd_denoising_steps,
                                     dtype=torch.long).cpu()
            if fastvideo_args.pipeline_config.warp_denoising_step:
                scheduler_timesteps = torch.cat(
                    (self.scheduler.timesteps.cpu(),
                     torch.tensor([0], dtype=torch.float32)))
                timesteps = scheduler_timesteps[1000 - timesteps]
            timesteps = timesteps.to(get_local_torch_device())

        boundary_ratio = getattr(fastvideo_args.pipeline_config.dit_config,
                                 'boundary_ratio', None)
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        # directly set the kwarg.
        image_kwargs = {"encoder_hidden_states_image": image_embeds}
        pos_cond_kwargs: dict[str, Any] = {}

        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents
        b, c, t, h, w = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        kv_cache1 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                              dtype=target_dtype,
                                              device=latents.device)
        kv_cache2 = None
        if boundary_timestep is not None:
            kv_cache2 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                                  dtype=target_dtype,
                                                  device=latents.device)

        kv_cache_mouse = None
        kv_cache_keyboard = None
        if self.use_action_module:
            kv_cache_mouse, kv_cache_keyboard = self._initialize_action_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device)

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=257,  # 1 CLS + 256 patch tokens
            dtype=target_dtype,
            device=latents.device)

        if t % self.num_frame_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frame_per_block for causal denoising"
            )
        num_blocks = t // self.num_frame_per_block
        block_sizes = [self.num_frame_per_block] * num_blocks
        start_index = 0

        if boundary_timestep is not None:
            block_sizes[0] = 1

        # NOTE: MatrixGame does NOT process the first frame separately.
        # The first frame information is already encoded in batch.image_latent (cond_concat)
        # and will be used by the model via channel concatenation: torch.cat([x, cond_concat], dim=1)

        ctx = BlockProcessingContext(
            batch=batch,
            block_idx=0,
            start_index=0,
            kv_cache1=kv_cache1,
            kv_cache2=kv_cache2,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            crossattn_cache=crossattn_cache,
            timesteps=timesteps,
            block_sizes=block_sizes,
            noise_pool=None,
            fastvideo_args=fastvideo_args,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            boundary_timestep=boundary_timestep,
            high_noise_timesteps=high_noise_timesteps,
            use_scheduler_step=False,
            context_noise=getattr(fastvideo_args.pipeline_config,
                                  "context_noise", 0),
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
        )

        context_noise = getattr(fastvideo_args.pipeline_config, "context_noise",
                                0)

        with self.progress_bar(total=len(block_sizes) *
                               len(timesteps)) as progress_bar:
            for block_idx, current_num_frames in enumerate(block_sizes):
                ctx.block_idx = block_idx
                ctx.start_index = start_index
                current_latents = latents[:, :, start_index:start_index +
                                          current_num_frames, :, :]

                action_kwargs = self._prepare_action_kwargs(
                    batch, start_index, current_num_frames)

                current_latents = self._process_single_block(
                    current_latents=current_latents,
                    batch=batch,
                    start_index=start_index,
                    current_num_frames=current_num_frames,
                    timesteps=timesteps,
                    ctx=ctx,
                    action_kwargs=action_kwargs,
                    progress_bar=progress_bar,
                )

                latents[:, :, start_index:start_index +
                        current_num_frames, :, :] = current_latents

                # Update KV caches with clean context
                # Skip for scheduler.step() models — they use re-noised
                # context prepending instead of KV cache for coherence
                if not ctx.use_scheduler_step:
                    self._update_context_cache(
                        current_latents=current_latents,
                        batch=batch,
                        start_index=start_index,
                        current_num_frames=current_num_frames,
                        ctx=ctx,
                        action_kwargs=action_kwargs,
                        context_noise=context_noise,
                    )

                start_index += current_num_frames

        if boundary_timestep is not None:
            num_frames_to_remove = self.num_frame_per_block - 1
            if num_frames_to_remove > 0:
                latents = latents[:, :, :-num_frames_to_remove, :, :]

        batch.latents = latents
        return batch

    def _prepare_action_kwargs(self, batch: ForwardBatch, start_index: int,
                               num_frames: int) -> dict[str, Any]:
        action_kwargs: dict[str, Any] = {}
        if not self.use_action_module:
            return action_kwargs

        vae_time_compression_ratio = 4
        end_frame_idx = 1 + vae_time_compression_ratio * (start_index +
                                                          num_frames - 1)
        if hasattr(batch, 'mouse_cond') and batch.mouse_cond is not None:
            action_kwargs['mouse_cond'] = batch.mouse_cond[:, :end_frame_idx]
        if hasattr(batch, 'keyboard_cond') and batch.keyboard_cond is not None:
            action_kwargs[
                'keyboard_cond'] = batch.keyboard_cond[:, :end_frame_idx]

        # CRITICAL: Pass num_frame_per_block to model - this should be num_frames (current block size)
        action_kwargs['num_frame_per_block'] = num_frames
        return action_kwargs

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype,
                             device: torch.device) -> list[dict]:
        kv_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = getattr(
            self.transformer, 'attention_head_dim',
            self.transformer.hidden_size // num_attention_heads)
        # WanGame's WanGameActionSelfAttention stores rope+prope keys
        # concatenated along dim=-1, so cache needs 2x head_dim
        if 'WanGame' in type(self.transformer).__name__:
            attention_head_dim = attention_head_dim * 2
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * self.sliding_window_num_frames

        per_layer_bytes = 2 * batch_size * kv_cache_size * num_attention_heads * attention_head_dim * 2  # k+v, bf16
        total_bytes = per_layer_bytes * self.num_transformer_blocks
        logger.info(
            "KV cache: layers=%d, cache_size=%d, heads=%d, head_dim=%d, "
            "per_layer=%.1f MiB, total=%.1f GiB",
            self.num_transformer_blocks, kv_cache_size,
            num_attention_heads, attention_head_dim,
            per_layer_bytes / 1024**2, total_bytes / 1024**3)

        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        return kv_cache

    def _initialize_action_kv_cache(self, batch_size: int, dtype: torch.dtype,
                                    device: torch.device):
        kv_cache_mouse = []
        kv_cache_keyboard = []

        action_heads = self.action_config.get('heads_num', 16)
        mouse_head_dim = self.action_config.get('mouse_hidden_dim',
                                                1024) // action_heads
        keyboard_head_dim = self.action_config.get('keyboard_hidden_dim',
                                                   1024) // action_heads

        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15

        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, action_heads, keyboard_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, action_heads, keyboard_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })
            kv_cache_mouse.append({
                "k":
                torch.zeros([
                    batch_size * self.frame_seq_length, kv_cache_size,
                    action_heads, mouse_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size * self.frame_seq_length, kv_cache_size,
                    action_heads, mouse_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        return kv_cache_mouse, kv_cache_keyboard

    def _initialize_crossattn_cache(self, batch_size: int, max_text_len: int,
                                    dtype: torch.dtype,
                                    device: torch.device) -> list[dict]:
        crossattn_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = getattr(
            self.transformer, 'attention_head_dim',
            self.transformer.hidden_size // num_attention_heads)
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([
                    batch_size, max_text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, max_text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "is_init":
                False,
            })
        return crossattn_cache

    def _denoise_one_step(
        self,
        current_latents: torch.Tensor,
        noise_latents_btchw: torch.Tensor,
        batch: ForwardBatch,
        start_index: int,
        current_num_frames: int,
        timestep: torch.Tensor,
        step_idx: int,
        next_timestep: torch.Tensor | None,
        ctx: BlockProcessingContext,
        action_kwargs: dict[str, Any],
        noise_generator: Callable[[tuple, torch.dtype, int], torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single denoising step (one timestep) for ORCA scheduling."""
        prompt_embeds = batch.prompt_embeds
        t_cur = timestep

        if ctx.boundary_timestep is not None and t_cur < ctx.boundary_timestep:
            current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
        else:
            current_model = self.transformer

        noise_latents = noise_latents_btchw.clone()
        latent_model_input = current_latents.to(ctx.target_dtype)

        independent_first_frame = getattr(self.transformer,
                                          'independent_first_frame', False)

        # For use_scheduler_step models: prepend previous blocks' denoised
        # frames (re-noised to current timestep) as temporal context.
        # This provides inter-block coherence without KV cache.
        context_num_frames = 0
        if ctx.use_scheduler_step and start_index > 0:
            # Get sigma for current timestep
            step_idx_for_sigma = self.scheduler.index_for_timestep(
                t_cur, self.scheduler.timesteps)
            sigma = self.scheduler.sigmas[step_idx_for_sigma]

            # Get previously denoised frames (clean, stored in batch.latents)
            # Use multiple blocks of context for better temporal coherence.
            # More context = better action following but more compute.
            max_context_frames = self.num_frame_per_block * 4  # 12 frames
            ctx_start = max(0, start_index - max_context_frames)
            context_clean = batch.latents[:, :, ctx_start:start_index].to(
                ctx.target_dtype)
            context_num_frames = context_clean.shape[2]

            # Re-noise context to match current timestep: x_t = (1-sigma)*x_0 + sigma*noise
            context_noise = torch.randn_like(context_clean)
            context_noised = (1 - sigma) * context_clean + sigma * context_noise

            # Prepend context to current block's latents (before image_latent concat)
            latent_model_input = torch.cat(
                [context_noised, latent_model_input], dim=2)

        # Concatenate image_latent along channel dim for ALL frames (context + current)
        if batch.image_latent is not None and independent_first_frame and start_index == 0:
            latent_model_input = torch.cat([
                latent_model_input,
                batch.image_latent.to(ctx.target_dtype)
            ],
                                           dim=2)
        elif batch.image_latent is not None and not independent_first_frame:
            # Slice image_latent to cover context + current frames
            img_start = start_index - context_num_frames
            img_end = start_index + current_num_frames
            img_lat = batch.image_latent[:, :, img_start:img_end, :, :]
            latent_model_input = torch.cat(
                [latent_model_input,
                 img_lat.to(ctx.target_dtype)], dim=1)

        total_frames = current_num_frames + context_num_frames

        t_expand = t_cur.repeat(latent_model_input.shape[0] *
                                current_num_frames)

        attn_metadata = None

        with torch.autocast(device_type="cuda",
                            dtype=ctx.target_dtype,
                            enabled=ctx.autocast_enabled), \
            set_forward_context(current_timestep=step_idx,
                                attn_metadata=attn_metadata,
                                forward_batch=batch):
            # Use per-frame timestep for causal models (e.g. MatrixGame),
            # but scalar timestep for models with action modules (e.g. WanGame)
            # that expand temb internally per-frame.
            if ctx.use_scheduler_step:
                # WanGame-style: pass [B] scalar timestep
                t_for_model = t_cur.repeat(latent_model_input.shape[0])
            else:
                # MatrixGame-style: pass [B, num_frames] per-frame timestep
                t_for_model = t_cur * torch.ones(
                    (latent_model_input.shape[0], current_num_frames),
                    device=latent_model_input.device,
                    dtype=torch.long)

            # For use_scheduler_step models (WanGame): skip KV cache during
            # denoising to avoid mixing clean cached context with noisy current
            # tokens. Context is provided via re-noised frame prepending instead.
            if ctx.use_scheduler_step:
                model_kwargs = {
                    "current_start": 0 if context_num_frames > 0 else start_index * self.frame_seq_length,
                    "start_frame": start_index - context_num_frames if context_num_frames > 0 else start_index,
                }
            else:
                model_kwargs = {
                    "kv_cache": ctx.get_kv_cache(t_cur),
                    "crossattn_cache": ctx.crossattn_cache,
                    "current_start": start_index * self.frame_seq_length,
                    "start_frame": start_index,
                }

            if self.use_action_module and current_model == self.transformer:
                model_kwargs.update({
                    "kv_cache_mouse": ctx.kv_cache_mouse,
                    "kv_cache_keyboard": ctx.kv_cache_keyboard,
                })
                model_kwargs.update(action_kwargs)

            if not self.use_action_module and batch.mouse_cond is not None and batch.keyboard_cond is not None:
                from fastvideo.models.dits.hyworld.pose import process_custom_actions
                viewmats, intrinsics, action_labels = process_custom_actions(
                    batch.keyboard_cond, batch.mouse_cond)
                # Slice to include context + current block frames
                action_slice_start = start_index - context_num_frames
                action_slice_end = start_index + current_num_frames
                if step_idx == 0:
                    logger.info(
                        f"[Actions] block start_index={start_index}, "
                        f"action_slice=[{action_slice_start}:{action_slice_end}], "
                        f"total_viewmats={viewmats.shape[0]}, "
                        f"action_labels[slice]={action_labels[action_slice_start:action_slice_end].tolist()}, "
                        f"keyboard_cond nonzero frames={batch.keyboard_cond.abs().sum(-1).squeeze().nonzero().squeeze().tolist()}"
                    )
                viewmats = viewmats[action_slice_start:action_slice_end]
                intrinsics = intrinsics[action_slice_start:action_slice_end]
                action_labels = action_labels[action_slice_start:action_slice_end]
                camera_action_kwargs = self.prepare_extra_func_kwargs(
                    current_model.forward,
                    {
                        "viewmats": viewmats.unsqueeze(0).to(
                            get_local_torch_device(), dtype=ctx.target_dtype),
                        "Ks": intrinsics.unsqueeze(0).to(
                            get_local_torch_device(), dtype=ctx.target_dtype),
                        "action": action_labels.unsqueeze(0).to(
                            get_local_torch_device(), dtype=ctx.target_dtype),
                    },
                )
                model_kwargs.update(camera_action_kwargs)

            pred_noise_btchw = current_model(
                latent_model_input,
                prompt_embeds,
                t_for_model,
                **ctx.image_kwargs,
                **ctx.pos_cond_kwargs,
                **model_kwargs,
            ).permute(0, 2, 1, 3, 4)

        if ctx.use_scheduler_step:
            # Slice to keep only current block's prediction (remove context frames)
            if context_num_frames > 0:
                pred_noise_btchw = pred_noise_btchw[:, context_num_frames:]
            noise_pred_bcthw = pred_noise_btchw.permute(0, 2, 1, 3, 4)
            if step_idx == 0:
                logger.info(
                    "Denoise step0: t_cur=%s, input_mean=%.4f, input_std=%.4f, "
                    "pred_mean=%.4f, pred_std=%.4f, context_frames=%d, "
                    "scheduler_sigmas[:5]=%s",
                    t_cur.item(), current_latents.mean().item(),
                    current_latents.std().item(),
                    noise_pred_bcthw.mean().item(), noise_pred_bcthw.std().item(),
                    context_num_frames,
                    self.scheduler.sigmas[:5].tolist())
            current_latents = self.scheduler.step(
                noise_pred_bcthw, t_cur, current_latents,
                return_dict=False)[0]
            if step_idx == 0 or step_idx == len(ctx.timesteps) - 1:
                logger.info(
                    "After step%d: output_mean=%.4f, output_std=%.4f, "
                    "sigma_cur=%.6f, sigma_next=%.6f",
                    step_idx,
                    current_latents.mean().item(), current_latents.std().item(),
                    self.scheduler.sigmas[step_idx].item(),
                    self.scheduler.sigmas[min(step_idx + 1, len(self.scheduler.sigmas) - 1)].item())
            noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        else:
            if ctx.boundary_timestep is not None and t_cur >= ctx.boundary_timestep:
                pred_video_btchw = pred_noise_to_x_bound(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=t_expand,
                    boundary_timestep=torch.ones_like(t_expand) *
                    ctx.boundary_timestep,
                    scheduler=self.scheduler).unflatten(
                        0, pred_noise_btchw.shape[:2])
            else:
                pred_video_btchw = pred_noise_to_pred_video(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=t_expand,
                    scheduler=self.scheduler).unflatten(
                        0, pred_noise_btchw.shape[:2])

            if next_timestep is not None:
                next_t = next_timestep * torch.ones(
                    [1], dtype=torch.long, device=pred_video_btchw.device)

                if noise_generator is not None:
                    noise = noise_generator(pred_video_btchw.shape,
                                            pred_video_btchw.dtype, step_idx)
                else:
                    noise = torch.randn(
                        pred_video_btchw.shape,
                        dtype=pred_video_btchw.dtype,
                        generator=(batch.generator[0] if isinstance(
                            batch.generator, list) else batch.generator)).to(
                                pred_video_btchw.device)

                noise_btchw = noise
                if ctx.boundary_timestep is not None and ctx.high_noise_timesteps is not None and step_idx < len(
                        ctx.high_noise_timesteps) - 1:
                    noise_latents_btchw = self.scheduler.add_noise_high(
                        pred_video_btchw.flatten(0, 1),
                        noise_btchw.flatten(0, 1), next_t,
                        torch.ones_like(next_t) *
                        ctx.boundary_timestep).unflatten(
                            0, pred_video_btchw.shape[:2])
                elif ctx.boundary_timestep is not None and ctx.high_noise_timesteps is not None and step_idx == len(
                        ctx.high_noise_timesteps) - 1:
                    noise_latents_btchw = pred_video_btchw
                else:
                    noise_latents_btchw = self.scheduler.add_noise(
                        pred_video_btchw.flatten(0, 1),
                        noise_btchw.flatten(0, 1),
                        next_t).unflatten(0, pred_video_btchw.shape[:2])
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
            else:
                current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)

        return current_latents, noise_latents_btchw

    def _process_single_block(
        self,
        current_latents: torch.Tensor,
        batch: ForwardBatch,
        start_index: int,
        current_num_frames: int,
        timesteps: torch.Tensor,
        ctx: BlockProcessingContext,
        action_kwargs: dict[str, Any],
        noise_generator: Callable[[tuple, torch.dtype, int], torch.Tensor]
        | None = None,
        progress_bar: Any | None = None,
    ) -> torch.Tensor:
        prompt_embeds = batch.prompt_embeds
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)

        for i, t_cur in enumerate(timesteps):
            if ctx.boundary_timestep is not None and t_cur < ctx.boundary_timestep:
                current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
            else:
                current_model = self.transformer

            noise_latents = noise_latents_btchw.clone()
            latent_model_input = current_latents.to(ctx.target_dtype)

            independent_first_frame = getattr(self.transformer,
                                              'independent_first_frame', False)
            if batch.image_latent is not None and independent_first_frame and start_index == 0:
                latent_model_input = torch.cat([
                    latent_model_input,
                    batch.image_latent.to(ctx.target_dtype)
                ],
                                               dim=2)
            elif batch.image_latent is not None and not independent_first_frame:
                # WanGame-style: concat image_latent along channel dim
                # image_latent shape: [B, C_img, T_total, H, W] — slice to current block
                img_lat = batch.image_latent[:, :, start_index:start_index +
                                             current_num_frames, :, :]
                latent_model_input = torch.cat(
                    [latent_model_input,
                     img_lat.to(ctx.target_dtype)], dim=1)

            # t_expand needs to be [batch * frames] to match flattened pred_noise/noise_latents
            t_expand = t_cur.repeat(latent_model_input.shape[0] *
                                    current_num_frames)

            # Build attention metadata if VSA is available
            if vsa_available and self.attn_backend == VideoSparseAttentionBackend:
                self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                )
                if self.attn_metadata_builder_cls is not None:
                    self.attn_metadata_builder = self.attn_metadata_builder_cls(
                    )
                    h, w = current_latents.shape[-2:]
                    attn_metadata = self.attn_metadata_builder.build(
                        current_timestep=i,
                        raw_latent_shape=(current_num_frames, h, w),
                        patch_size=ctx.fastvideo_args.pipeline_config.
                        dit_config.patch_size,
                        STA_param=batch.STA_param,
                        VSA_sparsity=ctx.fastvideo_args.VSA_sparsity,
                        device=get_local_torch_device(),
                    )
                    assert attn_metadata is not None, "attn_metadata cannot be None"
                else:
                    attn_metadata = None
            else:
                attn_metadata = None

            with torch.autocast(device_type="cuda",
                                dtype=ctx.target_dtype,
                                enabled=ctx.autocast_enabled), \
                set_forward_context(current_timestep=i,
                                    attn_metadata=attn_metadata,
                                    forward_batch=batch):
                # Use per-frame timestep for causal models (e.g. MatrixGame),
                # but scalar timestep for models with action modules (e.g. WanGame)
                if ctx.use_scheduler_step:
                    t_expanded_noise = t_cur.repeat(latent_model_input.shape[0])
                else:
                    t_expanded_noise = t_cur * torch.ones(
                        (latent_model_input.shape[0], current_num_frames),
                        device=latent_model_input.device,
                        dtype=torch.long)

                model_kwargs = {
                    "kv_cache": ctx.get_kv_cache(t_cur),
                    "crossattn_cache": ctx.crossattn_cache,
                    "current_start": start_index * self.frame_seq_length,
                    "start_frame": start_index,
                }

                if self.use_action_module and current_model == self.transformer:
                    model_kwargs.update({
                        "kv_cache_mouse":
                        ctx.kv_cache_mouse,
                        "kv_cache_keyboard":
                        ctx.kv_cache_keyboard,
                    })
                    model_kwargs.update(action_kwargs)

                # WanGame-style action kwargs (viewmats/Ks/action via process_custom_actions)
                if not self.use_action_module and batch.mouse_cond is not None and batch.keyboard_cond is not None:
                    from fastvideo.models.dits.hyworld.pose import process_custom_actions
                    viewmats, intrinsics, action_labels = process_custom_actions(
                        batch.keyboard_cond, batch.mouse_cond)
                    camera_action_kwargs = self.prepare_extra_func_kwargs(
                        current_model.forward,
                        {
                            "viewmats": viewmats.unsqueeze(0).to(
                                get_local_torch_device(),
                                dtype=ctx.target_dtype),
                            "Ks": intrinsics.unsqueeze(0).to(
                                get_local_torch_device(),
                                dtype=ctx.target_dtype),
                            "action": action_labels.unsqueeze(0).to(
                                get_local_torch_device(),
                                dtype=ctx.target_dtype),
                        },
                    )
                    model_kwargs.update(camera_action_kwargs)

                pred_noise_btchw = current_model(
                    latent_model_input,
                    prompt_embeds,
                    t_expanded_noise,
                    **ctx.image_kwargs,
                    **ctx.pos_cond_kwargs,
                    **model_kwargs,
                ).permute(0, 2, 1, 3, 4)

            if ctx.use_scheduler_step:
                # Standard scheduler path (WanGame): scheduler handles the step
                # pred_noise_btchw is [B, T, C, H, W], scheduler expects [B, C, T, H, W]
                noise_pred_bcthw = pred_noise_btchw.permute(0, 2, 1, 3, 4)
                current_latents = self.scheduler.step(
                    noise_pred_bcthw, t_cur, current_latents,
                    return_dict=False)[0]
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
            else:
                # DMD path: manual pred_noise_to_pred_video + add_noise
                if ctx.boundary_timestep is not None and t_cur >= ctx.boundary_timestep:
                    pred_video_btchw = pred_noise_to_x_bound(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        boundary_timestep=torch.ones_like(t_expand) *
                        ctx.boundary_timestep,
                        scheduler=self.scheduler).unflatten(
                            0, pred_noise_btchw.shape[:2])
                else:
                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler).unflatten(
                            0, pred_noise_btchw.shape[:2])

                if i < len(timesteps) - 1:
                    next_timestep = timesteps[i + 1] * torch.ones(
                        [1], dtype=torch.long, device=pred_video_btchw.device)

                    # Use custom noise generator if provided (for streaming), else generate
                    if noise_generator is not None:
                        noise = noise_generator(pred_video_btchw.shape,
                                                pred_video_btchw.dtype, i)
                    else:
                        noise = torch.randn(
                            pred_video_btchw.shape,
                            dtype=pred_video_btchw.dtype,
                            generator=(batch.generator[0] if isinstance(
                                batch.generator, list) else batch.generator)).to(
                                    pred_video_btchw.device)

                    noise_btchw = noise
                    if ctx.boundary_timestep is not None and ctx.high_noise_timesteps is not None and i < len(
                            ctx.high_noise_timesteps) - 1:
                        noise_latents_btchw = self.scheduler.add_noise_high(
                            pred_video_btchw.flatten(0, 1),
                            noise_btchw.flatten(0, 1), next_timestep,
                            torch.ones_like(next_timestep) *
                            ctx.boundary_timestep).unflatten(
                                0, pred_video_btchw.shape[:2])
                    elif ctx.boundary_timestep is not None and ctx.high_noise_timesteps is not None and i == len(
                            ctx.high_noise_timesteps) - 1:
                        noise_latents_btchw = pred_video_btchw
                    else:
                        noise_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0,
                                                     1), noise_btchw.flatten(0, 1),
                            next_timestep).unflatten(0, pred_video_btchw.shape[:2])
                    current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                else:
                    current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

            if progress_bar is not None:
                progress_bar.update()

        return current_latents

    def _update_context_cache(
        self,
        current_latents: torch.Tensor,
        batch: ForwardBatch,
        start_index: int,
        current_num_frames: int,
        ctx: BlockProcessingContext,
        action_kwargs: dict[str, Any],
        context_noise: float,
    ) -> None:
        prompt_embeds = batch.prompt_embeds
        latents_device = current_latents.device

        # Use per-frame timestep for causal models, scalar for WanGame-style
        if ctx.use_scheduler_step:
            t_context = torch.ones([current_latents.shape[0]],
                                   device=latents_device,
                                   dtype=torch.long) * int(context_noise)
        else:
            t_context = torch.ones(
                [current_latents.shape[0], current_num_frames],
                device=latents_device,
                dtype=torch.long) * int(context_noise)
        context_bcthw = current_latents.to(ctx.target_dtype)

        # Concat image_latent for WanGame-style models (channel dim)
        independent_first_frame = getattr(self.transformer,
                                          'independent_first_frame', False)
        if batch.image_latent is not None and not independent_first_frame:
            img_lat = batch.image_latent[:, :, start_index:start_index +
                                         current_num_frames, :, :]
            context_bcthw = torch.cat(
                [context_bcthw, img_lat.to(ctx.target_dtype)], dim=1)

        with torch.autocast(device_type="cuda",
                            dtype=ctx.target_dtype,
                            enabled=ctx.autocast_enabled), \
            set_forward_context(current_timestep=0,
                                attn_metadata=None,
                                forward_batch=batch):

            context_model_kwargs = {
                "kv_cache": ctx.kv_cache1,
                "crossattn_cache": ctx.crossattn_cache,
                "current_start": start_index * self.frame_seq_length,
                "start_frame": start_index,
            }

            # WanGame needs is_cache=True to write to KV cache
            is_cache_kwargs = self.prepare_extra_func_kwargs(
                self.transformer.forward, {"is_cache": True})
            context_model_kwargs.update(is_cache_kwargs)

            if self.use_action_module:
                context_model_kwargs.update({
                    "kv_cache_mouse":
                    ctx.kv_cache_mouse,
                    "kv_cache_keyboard":
                    ctx.kv_cache_keyboard,
                })
                context_model_kwargs.update(action_kwargs)

            # WanGame-style action kwargs (viewmats/Ks/action)
            if not self.use_action_module and batch.mouse_cond is not None and batch.keyboard_cond is not None:
                from fastvideo.models.dits.hyworld.pose import process_custom_actions
                viewmats, intrinsics, action_labels = process_custom_actions(
                    batch.keyboard_cond, batch.mouse_cond)
                # Slice to current block's frames
                viewmats = viewmats[start_index:start_index + current_num_frames]
                intrinsics = intrinsics[start_index:start_index + current_num_frames]
                action_labels = action_labels[start_index:start_index + current_num_frames]
                camera_action_kwargs = self.prepare_extra_func_kwargs(
                    self.transformer.forward,
                    {
                        "viewmats": viewmats.unsqueeze(0).to(
                            get_local_torch_device(),
                            dtype=ctx.target_dtype),
                        "Ks": intrinsics.unsqueeze(0).to(
                            get_local_torch_device(),
                            dtype=ctx.target_dtype),
                        "action": action_labels.unsqueeze(0).to(
                            get_local_torch_device(),
                            dtype=ctx.target_dtype),
                    },
                )
                context_model_kwargs.update(camera_action_kwargs)

            if ctx.boundary_timestep is not None and self.transformer_2 is not None:
                self.transformer_2(
                    context_bcthw,
                    prompt_embeds,
                    t_context,
                    kv_cache=ctx.kv_cache2,
                    crossattn_cache=ctx.crossattn_cache,
                    current_start=start_index * self.frame_seq_length,
                    start_frame=start_index,
                    **ctx.image_kwargs,
                    **ctx.pos_cond_kwargs,
                )

            self.transformer(
                context_bcthw,
                prompt_embeds,
                t_context,
                **ctx.image_kwargs,
                **ctx.pos_cond_kwargs,
                **context_model_kwargs,
            )

    def streaming_reset(self, batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_size = self.transformer.patch_size
        patch_ratio = patch_size[-1] * patch_size[-2]
        self.frame_seq_length = latent_seq_length // patch_ratio

        dmd_denoising_steps = getattr(fastvideo_args.pipeline_config,
                                       'dmd_denoising_steps', None)
        use_scheduler_step = dmd_denoising_steps is None

        if use_scheduler_step:
            # Standard scheduler path (e.g. WanGame): use scheduler.set_timesteps
            self.scheduler.set_timesteps(batch.num_inference_steps,
                                         device=get_local_torch_device())
            timesteps = self.scheduler.timesteps
        else:
            # DMD path: use explicit denoising steps
            timesteps = torch.tensor(dmd_denoising_steps,
                                     dtype=torch.long).cpu()
            if fastvideo_args.pipeline_config.warp_denoising_step:
                scheduler_timesteps = torch.cat(
                    (self.scheduler.timesteps.cpu(),
                     torch.tensor([0], dtype=torch.float32)))
                timesteps = scheduler_timesteps[1000 - timesteps]
            timesteps = timesteps.to(get_local_torch_device())

        boundary_ratio = getattr(fastvideo_args.pipeline_config.dit_config,
                                 'boundary_ratio', None)
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        # directly set the kwarg.
        image_kwargs = {"encoder_hidden_states_image": image_embeds}
        pos_cond_kwargs: dict[str, Any] = {}

        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, fastvideo_args)

        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents
        b, c, t, h, w = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        # Initialize caches
        kv_cache1 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                              dtype=target_dtype,
                                              device=latents.device)
        kv_cache2 = None
        if boundary_timestep is not None:
            kv_cache2 = self._initialize_kv_cache(batch_size=latents.shape[0],
                                                  dtype=target_dtype,
                                                  device=latents.device)

        kv_cache_mouse = None
        kv_cache_keyboard = None
        if self.use_action_module:
            kv_cache_mouse, kv_cache_keyboard = self._initialize_action_kv_cache(
                batch_size=latents.shape[0],
                dtype=target_dtype,
                device=latents.device)

        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=257,  # 1 CLS + 256 patch tokens
            dtype=target_dtype,
            device=latents.device)

        # Calculate block sizes
        if t % self.num_frame_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frame_per_block for causal denoising"
            )
        num_blocks = t // self.num_frame_per_block
        block_sizes = [self.num_frame_per_block] * num_blocks
        if boundary_timestep is not None:
            block_sizes[0] = 1

        # Pre-allocate noise pool
        num_denoising_steps = len(timesteps)
        noise_shape = (b, self.num_frame_per_block, c, h, w)
        noise_pool = [
            torch.randn(
                noise_shape,
                dtype=target_dtype,
                device=latents.device,
            ) for _ in range(num_denoising_steps - 1)
        ]

        # Create and store context
        self._streaming_ctx = BlockProcessingContext(
            batch=batch,
            block_idx=0,
            start_index=0,
            kv_cache1=kv_cache1,
            kv_cache2=kv_cache2,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            crossattn_cache=crossattn_cache,
            timesteps=timesteps,
            block_sizes=block_sizes,
            noise_pool=noise_pool,
            fastvideo_args=fastvideo_args,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            boundary_timestep=boundary_timestep,
            high_noise_timesteps=high_noise_timesteps,
            use_scheduler_step=use_scheduler_step,
            context_noise=getattr(fastvideo_args.pipeline_config,
                                  "context_noise", 0),
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
        )

        self._streaming_initialized = True
        return batch

    def streaming_step(
            self,
            keyboard_action: torch.Tensor | None = None,
            mouse_action: torch.Tensor | None = None) -> ForwardBatch:
        if not self._streaming_initialized or self._streaming_ctx is None:
            raise RuntimeError(
                "Streaming not initialized! Call streaming_reset first.")

        ctx = self._streaming_ctx
        if ctx.block_idx >= len(ctx.block_sizes):
            return ctx.batch

        batch = ctx.batch
        latents = batch.latents
        assert latents is not None, "latents must be set in batch"

        current_num_frames = ctx.block_sizes[ctx.block_idx]
        start_index = ctx.start_index

        current_latents = latents[:, :, start_index:start_index +
                                  current_num_frames, :, :]

        # Update batch with new actions for this block
        if keyboard_action is not None or mouse_action is not None:
            vae_ratio = 4
            start_frame = 0 if start_index == 0 else 1 + vae_ratio * (
                start_index - 1)

            if keyboard_action is not None:
                n = keyboard_action.shape[1]
                batch.keyboard_cond[:, start_frame:start_frame +
                                    n] = keyboard_action.to(
                                        batch.keyboard_cond.device)
            if mouse_action is not None:
                n = mouse_action.shape[1]
                batch.mouse_cond[:, start_frame:start_frame +
                                 n] = mouse_action.to(batch.mouse_cond.device)

        action_kwargs = self._prepare_action_kwargs(batch, start_index,
                                                    current_num_frames)

        # Create noise generator that uses pre-allocated noise pool
        def streaming_noise_generator(shape: tuple, dtype: torch.dtype,
                                      step_idx: int) -> torch.Tensor:
            if ctx.noise_pool is not None and step_idx < len(ctx.noise_pool):
                return ctx.noise_pool[step_idx][:, :shape[1], :, :, :].to(
                    latents.device)
            else:
                # Fallback to dynamic allocation if pool not available
                return torch.randn(
                    shape,
                    dtype=dtype,
                    generator=(batch.generator[0] if isinstance(
                        batch.generator, list) else batch.generator)).to(
                            latents.device)

        current_latents = self._process_single_block(
            current_latents=current_latents,
            batch=batch,
            start_index=start_index,
            current_num_frames=current_num_frames,
            timesteps=ctx.timesteps,
            ctx=ctx,
            action_kwargs=action_kwargs,
            noise_generator=streaming_noise_generator,
        )

        latents[:, :, start_index:start_index +
                current_num_frames, :, :] = current_latents

        # Update KV caches with clean context
        # Skip for scheduler.step() models — they use re-noised
        # context prepending instead of KV cache for coherence
        if not ctx.use_scheduler_step:
            self._update_context_cache(
                current_latents=current_latents,
                batch=batch,
                start_index=start_index,
                current_num_frames=current_num_frames,
                ctx=ctx,
                action_kwargs=action_kwargs,
                context_noise=ctx.context_noise,
            )

        # Advance streaming state
        ctx.start_index += current_num_frames
        ctx.block_idx += 1

        return batch

    def streaming_clear(self) -> None:
        self._streaming_initialized = False
        self._streaming_ctx = None

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check("image_latent", batch.image_latent,
                         V.none_or_tensor_with_dims(5))
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale,
                         V.positive_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, lambda x:
            not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result
