"""Statically retained original cached-AV parser; see source_manifest.json."""
import torch
from torch.utils.data import Sampler
from contextlib import nullcontext
from .models.minimax_h3_dit import pack_audio, patchify_video
from .packing import MiniMaxH3Unit_PackedSequenceBuilder


def require_tensor(value, name, ndim, last_dim=None):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D tensor")
    if not value.is_floating_point() or any(n == 0 for n in value.shape):
        raise ValueError(f"{name} must be a nonempty floating-point tensor")
    if last_dim is not None and value.shape[-1] != last_dim:
        raise ValueError(f"{name} must have last dimension {last_dim}, got {value.shape}")
    return value


def check_video(value, name):
    value = require_tensor(value, name, 5)
    if value.shape[:2] != (1, 24) or any(n % 2 for n in value.shape[-2:]):
        raise ValueError(f"{name} must be [1,24,T,H,W] with even H/W, got {value.shape}")
    return value


def check_audio(value, name):
    value = require_tensor(value, name, 3)
    if value.shape[:2] != (2, 32):
        raise ValueError(f"{name} must be stereo [2,32,T], got {value.shape}")
    return value


def check_clean_anchor(shared, anchor, noise_key):
    """The repository's reference encoder casts the clean coefficient to latent dtype."""
    coefficient = float(shared.get(noise_key, 1.0))
    if not 0 <= coefficient <= 1:
        raise ValueError(f"Invalid {noise_key}={coefficient}")
    effective = torch.tensor(coefficient, dtype=anchor.dtype).item()
    if effective != 1.0:
        raise ValueError(
            f"Cached reference is already noise-augmented: {noise_key}={coefficient}, "
            f"effective coefficient={effective} in {anchor.dtype}. Re-cache clean references "
            "with coefficient 1.0, or provide clean ref_video_latents/ref_audio_latents. "
            "Changing a timestep cannot undo noise already stored in a cache."
        )


def reference_shape_from_packed(packed, visual_rows):
    """Recover the single previous chunk's grid without assuming target resolution."""
    if not isinstance(packed, dict):
        raise ValueError("Packed ref_visual_anchor requires cached packed positions; otherwise use ref_video_latents")
    positions = packed["img_pos"][:visual_rows].long()
    grid = packed["img_position_ids"][0, positions].double()
    latent_t, patch_h, patch_w = (int(grid[:, axis].unique().numel()) for axis in range(3))
    if latent_t * patch_h * patch_w != visual_rows:
        raise ValueError("Reference anchors must describe one rectangular previous video chunk")
    # Verify row order as well as counts: patchify_video is time/height/width major.
    expected = torch.cartesian_prod(*(grid[:, axis].unique(sorted=True) for axis in range(3)))
    if not torch.equal(grid, expected):
        raise ValueError("Reference video positions are not a single time/height/width grid")
    return latent_t, patch_h * 2, patch_w * 2


def prepare_cached_inputs(data):
    """Read official split-cache triples or flat tensor dicts; rebuild continuous RoPE.

    Only a single previous audio/video chunk is accepted as the reference. Existing
    cache flags, stale packed positions and keyframe/retake conditioning are not used.
    """
    if isinstance(data, (tuple, list)) and len(data) == 3:
        shared, positive, _negative = data
    elif isinstance(data, dict):
        shared, positive = data, data
    else:
        raise ValueError("Expected (inputs_shared, inputs_posi, inputs_nega) or a flat tensor dict")
    if not isinstance(shared, dict) or not isinstance(positive, dict):
        raise ValueError("Shared and positive cache entries must be dictionaries")

    video = check_video(shared.get("input_latents", shared.get("target_video_latents")), "input_latents")
    audio = check_audio(shared.get("audio_input_latents", shared.get("target_audio_latents")), "audio_input_latents")
    prompt = require_tensor(positive.get("prompt_embeds"), "prompt_embeds", 2, 5120)
    tags = positive.get("text_token_tags")
    if tags is None:
        tags = torch.ones(prompt.shape[0], dtype=torch.long)
    if not isinstance(tags, torch.Tensor) or tags.shape != (prompt.shape[0],):
        raise ValueError("text_token_tags must contain one tag per prompt token")
    if not torch.all(tags == 1):
        raise ValueError("This script expects FL2VA text-only prompt embeddings (all text_token_tags == 1)")

    for key in ("keyframe_cond_anchor", "input_latents_video", "input_latents_audio"):
        if shared.get(key) is not None:
            raise ValueError(f"{key} is not supported: expected T2AV or previous-chunk AV conditioning")

    raw_video, raw_audio = shared.get("ref_video_latents"), shared.get("ref_audio_latents")
    visual, sound = shared.get("ref_visual_anchor"), shared.get("ref_audio_anchor")
    if raw_video is not None or raw_audio is not None:
        raw_video = check_video(raw_video, "ref_video_latents")
        raw_audio = check_audio(raw_audio, "ref_audio_latents")
        ref_t, ref_h, ref_w = raw_video.shape[2:]
        visual, sound = patchify_video(raw_video), pack_audio(raw_audio)
    elif visual is not None or sound is not None:
        visual = require_tensor(visual, "ref_visual_anchor", 2, 96)
        sound = require_tensor(sound, "ref_audio_anchor", 2, 32)
        check_clean_anchor(shared, visual, "imgvid_cond_noise_aug")
        check_clean_anchor(shared, sound, "audio_cond_noise_aug")
        ref_t, ref_h, ref_w = reference_shape_from_packed(positive.get("packed"), len(visual))

    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    target_t, target_h, target_w = video.shape[2:]
    if visual is not None:
        if len(sound) % 2:
            raise ValueError("ref_audio_anchor must contain equal left/right channel lengths")
        block = dict(kind="video_audio", latent_t=ref_t, latent_h=ref_h,
                     latent_w=ref_w, ref_audio_t=len(sound) // 2)
        packed = builder._build_packed_ref2va(
            len(prompt), target_t, target_h, target_w, audio.shape[-1], [block],
        )
    else:
        packed = builder._build_packed_fl2va(
            len(prompt), target_t, target_h, target_w, audio.shape[-1], [],
        )
    packed["token_tags"][packed["text_pos"]] = tags.long()
    inputs = dict(
        input_latents=video, audio_input_latents=audio,
        prompt_embeds=prompt, packed=packed,
        # model_fn uses t=1-sigma: clean reference tokens must have t=1.
        imgvid_cond_noise_aug=1.0, audio_cond_noise_aug=1.0,
    )
    if visual is not None:
        inputs.update(ref_visual_anchor=visual, ref_audio_anchor=sound)
    return inputs


class NodeLocalSampler(Sampler):
    """Shard only over LOCAL ranks; cycle shorter nodes to equal optimizer steps."""
    def __init__(self, size, local_rank, local_world_size, num_samples, seed=42):
        if size <= 0 or not 0 <= local_rank < local_world_size or num_samples <= 0:
            raise ValueError("Invalid local sampler size/rank/num_samples")
        self.size, self.rank, self.world = size, local_rank, local_world_size
        self.num_samples, self.seed, self.epoch = num_samples, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        return (order[(self.rank + step * self.world) % self.size] for step in range(self.num_samples))


def gathered_parameter(parameter, modifier_rank=0):
    """All ranks enter; materialize only one ZeRO-3 parameter at a time."""
    if hasattr(parameter, "ds_id"):
        import deepspeed
        return deepspeed.zero.GatheredParameters([parameter], modifier_rank=modifier_rank)
    return nullcontext()
