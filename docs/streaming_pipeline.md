# Streaming Pipeline Architecture: MatrixGame & WanGame

## Overview

Both MatrixGame and WanGame use the same streaming infrastructure but differ in their denoising strategy. The key distinction is controlled by `use_scheduler_step`:

- **MatrixGame**: `use_scheduler_step=False` — uses DMD (Distribution Matching Distillation) with 3 fixed timesteps, KV cache for temporal coherence
- **WanGame**: `use_scheduler_step=True` — uses standard `scheduler.step()` with N inference steps, re-noised context prepending for temporal coherence

## System Architecture

```
Browser (WebSocket)
    │
FastAPI Server (main.py)
    │
GPUPool ── GPUSlot[0] ── subprocess ── StreamingVideoGenerator ── MultiprocExecutor
         │                                                              │
         ├─ GPUSlot[1] ── ...                              WorkerMultiprocProc
         │                                                              │
         └─ GPUSlot[N] ── ...                              multi_user_streaming_loop
                                                                        │
                                                               MultiUserEngine
                                                                        │
                                                    ┌───────────────────┼───────────────────┐
                                                    │                   │                   │
                                              _denoise_one_step   _finalize_block    _update_context_cache
                                                    │                   │
                                              Transformer          VAE Decode
```

### Multi-GPU: Horizontal Scaling
Each GPU runs an independent subprocess with its own full model copy (`num_gpus=1`). Users are co-located onto the same GPU to maximize ORCA batching, spilling to the next GPU only when `MAX_USERS_PER_GPU` (16) is reached.

### Multi-User on a Single GPU: ORCA Scheduling
The `MultiUserEngine` batches users at the denoising-step level. Users at the same `(block_idx, denoising_step)` are grouped and run through the transformer in a single forward pass.

## Pipeline Stages

Both models use `MatrixGameCausalDMDPipeline`, which defines these stages:

```
input_validation → prompt_encoding → image_encoding → conditioning
    → latent_preparation → image_latent_preparation → denoising → decoding
```

### Stage: image_latent_preparation (`MatrixGameImageVAEEncodingStage`)

Creates the `image_latent` (cond_concat) tensor `[B, 20, T_latent, H, W]`:
1. Constructs a video tensor: `[first_frame, zeros × (num_frames-1)]`
2. Encodes the **full video** through the VAE (important: temporal convolutions spread information across all latent frames; encoding a truncated video produces incorrect latents)
3. Normalizes using `latents_mean`/`latents_std`
4. Creates `mask_cond` (ones for first frame, zeros for rest)
5. Concatenates: `cond_concat = [mask_cond[:, :4], img_cond]` → 4 + 16 = 20 channels

### Stage: denoising (`MatrixGameCausalDenoisingStage`)

This single stage handles both MatrixGame and WanGame via `use_scheduler_step` branching.

## Streaming Flow

### Initialization: `streaming_reset()`

Called once per user when they join.

1. Determines `use_scheduler_step`:
   - `dmd_denoising_steps` is set → MatrixGame DMD path (`use_scheduler_step=False`)
   - `dmd_denoising_steps` is None → WanGame scheduler path (`use_scheduler_step=True`)
2. Computes timesteps:
   - **MatrixGame**: `[1000, 666, 333]` → warped via scheduler → 3 actual timesteps
   - **WanGame**: `scheduler.set_timesteps(num_inference_steps)` → N timesteps
3. Initializes KV caches (for all models), noise pool, block sizes
4. Stores everything in `BlockProcessingContext`

### Per-Block Generation: `submit_step()` → `run_iteration()` → `_denoise_one_step()` × N

#### 1. `submit_step(user_id, keyboard_action, mouse_action)`
- Writes new actions into `batch.keyboard_cond` / `batch.mouse_cond`
- Prepares `action_kwargs` (sliced mouse/keyboard tensors + `num_frame_per_block`)
- Sets `session.denoising_step = 0` to mark the user as having pending work

#### 2. `run_iteration()` (ORCA scheduler)
- Groups users by `(block_idx, denoising_step)`
- For each group, batches their tensors and calls `_denoise_one_step()`
- Unbatches results back to individual users
- When a user completes all denoising steps → calls `_finalize_block()`

#### 3. `_denoise_one_step()` — The Core Divergence Point

**Common to both paths:**
- Selects model (`transformer` or `transformer_2` based on boundary_timestep)
- Clones `noise_latents_btchw`

**Then branches:**

---

## MatrixGame DMD Path (`use_scheduler_step=False`)

### Image Latent Handling
MatrixGame's `CausalMatrixGameWanModel` has `_concatenates_image_latent = True`. The model grabs `batch.image_latent` from `forward_context` and concatenates it internally along the channel dim (16 noise + 20 image = 36 channels). The denoising stage skips external concatenation.

### Timestep Format
Per-frame: `[B, num_frames]` — e.g., `[[1000, 1000, 1000]]` for 3 latent frames.

### Model Forward
```python
model_kwargs = {
    "kv_cache": ctx.get_kv_cache(t_cur),    # KV cache for temporal attention
    "crossattn_cache": ctx.crossattn_cache,  # Cross-attention cache for text
    "current_start": start_index * frame_seq_length,
    "start_frame": start_index,
    "kv_cache_mouse": ...,                   # Action module caches
    "kv_cache_keyboard": ...,
    "mouse_cond": ...,                       # Sliced action tensors
    "keyboard_cond": ...,
    "num_frame_per_block": num_frames,
}
pred_noise = model(latent_model_input, prompt_embeds, timestep, **model_kwargs)
```

### Denoising Logic (3 steps)
```
For each timestep t in [t0, t1, t2]:
    pred_noise = model(noisy_latents, t)
    pred_video = pred_noise_to_pred_video(pred_noise, noisy_latents, t)
    if not last step:
        noisy_latents = scheduler.add_noise(pred_video, fresh_noise, t_next)
    else:
        output = pred_video  # Final clean prediction
```

Uses pre-allocated `noise_pool` for deterministic noise across blocks.

### Temporal Coherence: KV Cache
After each block completes, `_update_context_cache()` runs the **clean** denoised latents through the transformer with `context_noise` timestep (typically 0), writing into `kv_cache1`. Subsequent blocks attend to these cached keys/values during their denoising steps.

```
Block 0: denoise → output → update_context_cache (write to KV)
Block 1: denoise (attend to KV from block 0) → output → update_context_cache (append to KV)
Block 2: denoise (attend to KV from blocks 0+1) → ...
```

The KV cache uses a sliding window (`local_attn_size` blocks) — old entries are evicted when the cache is full.

---

## WanGame Scheduler Path (`use_scheduler_step=True`)

### Image Latent Handling
WanGame does NOT set `_concatenates_image_latent`. The denoising stage concatenates `image_latent` externally along the channel dim before passing to the model:
```python
img_lat = batch.image_latent[:, :, start:end, :, :]
latent_model_input = torch.cat([latent_model_input, img_lat], dim=1)  # 16 + 20 = 36 channels
```

### Temporal Context: Re-noised Frame Prepending
Instead of KV cache, WanGame prepends previous blocks' clean latents (re-noised to the current timestep) as temporal context:
```python
if start_index > 0:
    sigma = scheduler.sigmas[step_idx]
    context_clean = batch.latents[:, :, ctx_start:start_index]  # Up to 4 blocks back
    context_noised = (1 - sigma) * context_clean + sigma * noise
    latent_model_input = torch.cat([context_noised, latent_model_input], dim=2)
```

### Timestep Format
Scalar: `[B]` — e.g., `[1000]`. WanGame's model expands this per-frame internally.

### Model Forward
```python
model_kwargs = {
    "current_start": 0,                     # Reset due to context prepending
    "start_frame": start_index - context_num_frames,
    # NO kv_cache — context provided via prepending instead
    "viewmats": ...,                         # Camera matrices for PRoPE
    "Ks": ...,                               # Intrinsics
    "action": ...,                           # Discrete action labels
}
```

### Denoising Logic (N steps via scheduler)
```
For each timestep t in scheduler.timesteps:
    pred_noise = model(latent_model_input, t)
    if context_frames > 0:
        pred_noise = pred_noise[:, context_frames:]  # Remove context predictions
    current_latents = scheduler.step(pred_noise, t, current_latents)
```

### Temporal Coherence: No KV Cache Update
`_update_context_cache()` is skipped for `use_scheduler_step=True` models. Coherence comes entirely from the re-noised context prepending.

---

## Key Differences Summary

| Aspect | MatrixGame (DMD) | WanGame (Scheduler) |
|---|---|---|
| `use_scheduler_step` | `False` | `True` |
| Denoising steps | 3 (fixed: 1000, 666, 333) | N (configurable, e.g. 10-40) |
| Step function | `pred_noise_to_pred_video` + `add_noise` | `scheduler.step()` |
| Temporal coherence | KV cache (`_update_context_cache`) | Re-noised context prepending |
| Image latent concat | Internal (model's `_forward_inference`) | External (`_denoise_one_step`) |
| `_concatenates_image_latent` | `True` | Not set (defaults to `False`) |
| Timestep format | Per-frame `[B, T]` | Scalar `[B]` |
| KV cache during denoising | Yes | No (omitted from model_kwargs) |
| Action module | `ActionModule` with mouse/keyboard KV caches | `process_custom_actions` → viewmats/Ks/action |
| Noise source | Pre-allocated noise pool | Random per step |

## Key Data Structures

### `BlockProcessingContext`
Holds all per-user streaming state:
- `kv_cache1`, `kv_cache2` — transformer KV caches `[B, cache_size, heads, dim]`
- `kv_cache_mouse`, `kv_cache_keyboard` — action module caches
- `crossattn_cache` — cross-attention cache for text embeddings
- `timesteps` — denoising timestep schedule
- `block_sizes` — number of latent frames per block (typically all 3)
- `noise_pool` — pre-allocated noise tensors for deterministic generation
- `use_scheduler_step` — which denoising path to use

### `ForwardBatch`
Holds the generation state:
- `latents` — `[B, C, T, H, W]` noise/denoised latents for all frames
- `image_latent` — `[B, 20, T_latent, H, W]` image conditioning (cond_concat)
- `prompt_embeds` — text embeddings
- `keyboard_cond` — `[B, num_raw_frames, keyboard_dim]` action tensors
- `mouse_cond` — `[B, num_raw_frames, 2]` mouse pitch/yaw deltas

## File Map

```
fastvideo/pipelines/stages/
├── matrixgame_denoising.py     # MatrixGameCausalDenoisingStage (both DMD + scheduler paths)
├── multi_user_engine.py        # MultiUserEngine (ORCA batching, _finalize_block)
├── image_encoding.py           # MatrixGameImageVAEEncodingStage (VAE encode → cond_concat)
├── denoising.py                # Base DenoisingStage
└── ...

fastvideo/pipelines/basic/matrixgame/
├── matrixgame_causal_dmd_pipeline.py  # Pipeline stage wiring

fastvideo/models/dits/matrixgame/
├── causal_model.py             # CausalMatrixGameWanModel (_concatenates_image_latent=True)
├── action_module.py            # ActionModule (mouse/keyboard attention)
└── ...

fastvideo/models/dits/wangame/
├── causal_model.py             # WanGameActionTransformer3DModel
└── ...

ui/world_model/server/
├── main.py                     # FastAPI server, WebSocket handling
├── gpu_pool.py                 # GPUPool, GPUSlot (multi-GPU management)
└── config.py                   # Model registry, NUM_INFERENCE_STEPS, MAX_USERS_PER_GPU
```
