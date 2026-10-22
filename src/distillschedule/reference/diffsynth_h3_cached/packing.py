"""Cached packing/model_fn closure, not the complete DiffSynth pipeline."""
from __future__ import annotations
import numpy as np
import torch
from .support import PipelineUnit
from .models.minimax_h3_dit import pack_audio, unpack_audio, patchify_video, unpatchify_video


class MiniMaxH3Unit_PackedSequenceBuilder(PipelineUnit):
    _INTERP = 32
    _T_GROUP = 5
    _FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    _FRAME_RESCALE = 5.0 / 3.0
    _SEQ_ALIGN = 64

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params_nega={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params=("video_latents", "audio_latents", "keyframe_cond_anchor", "keyframe_indices", "ref_blocks"),
            output_params=("packed",)
        )

    @staticmethod
    def _to_device(packed: dict, device) -> dict:
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in packed.items()}

    def _aligned_seq_len(self, used: int) -> int:
        return ((used + self._SEQ_ALIGN - 1) // self._SEQ_ALIGN) * self._SEQ_ALIGN

    def _axis_from_sqrt_area(self, dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * self._INTERP
        return torch.from_numpy(grid).to(torch.float64)

    def _video_t_grid(self, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor([self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    def _temporal_position_span(self, temporal_length: int) -> float:
        spans = np.ones(int(temporal_length), dtype=np.float64) * self._FRAME_RESCALE
        for token_index in range(self._T_GROUP):
            spans[token_index::self._T_GROUP] *= self._FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    def _video_t_span(self, n: int) -> float:
        return sum(self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n))

    def _frame_grid(self, latent_h: int, latent_w: int, sqrt_area):
        h_grid = self._axis_from_sqrt_area(latent_h, 2, sqrt_area)
        w_grid = self._axis_from_sqrt_area(latent_w, 2, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_grid

    def _video_grid(self, latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
        video_g = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
        video_g[:, :, 0] = self._video_t_grid(latent_t, origin)[:, None]
        video_g[:, :, 1:] = frame[None]
        return video_g.reshape(-1, 3)

    def _audio_w_axis(self, w_grid: torch.Tensor, audio_t: int, audio_rows: int) -> torch.Tensor:
        return torch.cat([torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
                          torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64)])

    def _build_packed_fl2va(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframe_indices, audio_channel=2):
        """fl2va layout: [text | cond | audio | video | pad]."""
        frame_rows = (latent_h // 2) * (latent_w // 2)
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        num_keyframes = len(keyframe_indices)
        cond_rows = num_keyframes * frame_rows
        used = text_len + cond_rows + audio_rows + video_rows
        seq_len = self._aligned_seq_len(used)

        text_sl = slice(0, text_len)
        cond_sl = slice(text_len, text_len + cond_rows)
        audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        # img_pos covers both cond AND video rows, conditions first
        img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

        sqrt_area = np.sqrt(latent_h * latent_w)
        frame, w_grid = self._frame_grid(latent_h, latent_w, sqrt_area)

        # Condition rows: temporal position depends on frame_index
        temporal_span = self._temporal_position_span(latent_t)
        for i, idx in enumerate(keyframe_indices):
            sl = slice(i * frame_rows, (i + 1) * frame_rows)
            if idx == 0:
                cond_t = float(text_len)
            else:  # idx == -1
                cond_t = float(text_len) + temporal_span - self._FRAME_RESCALE
            cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
            cond_g[:, 0] = cond_t
            cond_g[:, 1:] = frame
            g[cond_sl.start + sl.start: cond_sl.start + sl.stop] = cond_g

        g[video_sl] = self._video_grid(latent_t, frame, float(text_len))
        g[audio_sl, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
        g[audio_sl, 2] = self._audio_w_axis(w_grid, audio_t, audio_rows)

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0  # both cond and video rows are tagged as video (0)

        return {
            "img_pos": img_pos, "audio_pos": audio_pos, "text_pos": torch.arange(0, text_len),
            "img_position_ids": g[None], "token_tags": token_tags,
            "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32), "seq_len": seq_len,
        }

    def _build_packed_ref2va(self, text_len, latent_t, latent_h, latent_w, audio_t, ref_blocks, audio_channel=2):
        """ref2va layout: [text | ref_0 | ref_1 | ... | target_audio | target_video | pad]"""
        ph, pw = latent_h // 2, latent_w // 2
        target_frame_rows = ph * pw
        target_video_rows = latent_t * target_frame_rows
        target_audio_rows = audio_t * audio_channel

        block_dims, total_ref_visual_rows, total_ref_audio_rows = [], 0, 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind, "visual_rows": 0, "audio_rows": 0, "ref_audio_t": int(b.get("ref_audio_t", 0))}
            if kind in ("image", "video", "video_audio"):
                lt_r, lh_r, lw_r = int(b["latent_t"]), int(b["latent_h"]), int(b["latent_w"])
                info.update(visual_rows=lt_r * (lh_r // 2) * (lw_r // 2), latent_t=lt_r, latent_h=lh_r, latent_w=lw_r)
                total_ref_visual_rows += info["visual_rows"]
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            elif kind == "audio":
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            else:
                raise ValueError(f"unknown ref kind: {kind}")
            block_dims.append(info)

        used = text_len + total_ref_visual_rows + total_ref_audio_rows + target_audio_rows + target_video_rows
        seq_len = self._aligned_seq_len(used)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[0:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[0:text_len] = 1

        # Iterate through ref blocks, placing them contiguously.
        cursor, t_cursor = text_len, float(text_len)
        ref_visual_pos_parts, ref_audio_pos_parts = [], []
        # Target w_grid used for audio W-axis (channel separation)
        target_sqrt_area = float(np.sqrt(latent_h * latent_w))
        target_w_grid = self._axis_from_sqrt_area(latent_w, 2, target_sqrt_area)

        for info in block_dims:
            kind = info["kind"]
            if kind == "image":
                v_rows, lh_r, lw_r = info["visual_rows"], info["latent_h"], info["latent_w"]
                # Own spatial grid
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, w_grid = self._frame_grid(lh_r, lw_r, sqrt_area)
                sl = slice(cursor, cursor + v_rows)
                g[sl, 0] = t_cursor
                g[sl, 1:] = frame
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows
                t_cursor += 1.0

            elif kind in ("video", "video_audio"):
                a_rows, v_rows, ref_at = info["audio_rows"], info["visual_rows"], info["ref_audio_t"]
                lt_r, lh_r, lw_r = info["latent_t"], info["latent_h"], info["latent_w"]
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, rv_w_grid = self._frame_grid(lh_r, lw_r, sqrt_area)

                audio_sl = slice(cursor, cursor + a_rows)
                visual_sl = slice(audio_sl.stop, audio_sl.stop + v_rows)

                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[audio_sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    # W-axis uses THIS reference video's own grid, not the target's.
                    g[audio_sl, 2] = self._audio_w_axis(rv_w_grid, ref_at, a_rows)
                    token_tags[audio_sl] = 2
                    ref_audio_pos_parts.append(torch.arange(audio_sl.start, audio_sl.stop))

                g[visual_sl] = self._video_grid(lt_r, frame, t_cursor)
                token_tags[visual_sl] = 0
                ref_visual_pos_parts.append(torch.arange(visual_sl.start, visual_sl.stop))

                cursor = visual_sl.stop
                t_cursor += max(float(ref_at), self._video_t_span(lt_r))

            elif kind == "audio":
                a_rows, ref_at = info["audio_rows"], info["ref_audio_t"]
                sl = slice(cursor, cursor + a_rows)
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    g[sl, 2] = self._audio_w_axis(target_w_grid, ref_at, a_rows)
                    token_tags[sl] = 2
                    ref_audio_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += a_rows
                t_cursor += float(ref_at)

        # Target audio + target video after ref blocks
        target_audio_sl = slice(cursor, cursor + target_audio_rows)
        target_video_sl = slice(target_audio_sl.stop, target_audio_sl.stop + target_video_rows)

        # Target spatial grid (own)
        frame_t, w_grid_t = self._frame_grid(latent_h, latent_w, target_sqrt_area)
        g[target_video_sl] = self._video_grid(latent_t, frame_t, t_cursor)

        target_audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
        g[target_audio_sl, 0] = target_audio_t_grid.repeat(audio_channel)
        g[target_audio_sl, 2] = self._audio_w_axis(w_grid_t, audio_t, target_audio_rows)

        # Build combined img_pos / audio_pos
        target_video_pos = torch.arange(target_video_sl.start, target_video_sl.stop)
        target_audio_pos = torch.arange(target_audio_sl.start, target_audio_sl.stop)

        if ref_visual_pos_parts:
            img_pos = torch.cat(ref_visual_pos_parts + [target_video_pos])
        else:
            img_pos = target_video_pos
        if ref_audio_pos_parts:
            audio_pos = torch.cat(ref_audio_pos_parts + [target_audio_pos])
        else:
            audio_pos = target_audio_pos

        token_tags[img_pos] = 0
        token_tags[target_audio_sl] = 2

        text_pos = torch.arange(0, text_len)
        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)

        return {
            "img_pos": img_pos, "audio_pos": audio_pos, "text_pos": text_pos,
            "img_position_ids": g[None], "token_tags": token_tags, "cu_seqlens": cu,
            "seq_len": seq_len,
        }

    def process(self, pipe: MiniMaxH3Pipeline, prompt_embeds, video_latents, audio_latents, text_token_tags=None, keyframe_cond_anchor=None, keyframe_indices=None, ref_blocks=None):  # noqa: F821 - full-SFT pipeline is outside the cached-only closure
        text_len = prompt_embeds.shape[0]
        video_latent_t, latent_h, latent_w = video_latents.shape[2:]
        audio_latent_t = audio_latents.shape[-1]
        if ref_blocks is not None:
            packed = self._build_packed_ref2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t, ref_blocks)
        else:
            packed = self._build_packed_fl2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t, keyframe_indices if keyframe_cond_anchor is not None else [])

        packed["token_tags"][packed["text_pos"]] = text_token_tags.cpu()
        if pipe.device == "mps":
            packed["img_position_ids"] = packed["img_position_ids"].to(torch.float32)
        return {"packed": self._to_device(packed, pipe.device)}


def model_fn_minimax_h3_controlnet(
    dit,
    x,
    audio_x,
    img_position_ids,
    unique_timesteps,
    inverse_indices,
    token_tags,
    prompt_embeds,
    img_pos_info,
    audio_pos_info,
    text_pos_info,
    packed_seq_params,
    refiner_packed_seq_params,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    controlnet=None,
    control_rows=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_indices = inverse_indices.view(-1).to(torch.long)
    token_tags = token_tags.view(-1).to(torch.long)
    text_selected = prompt_embeds

    img_pos = img_pos_info["position_ids"].view(-1).to(torch.long)
    audio_pos = audio_pos_info["position_ids"].view(-1).to(torch.long)
    text_pos = text_pos_info["position_ids"].view(-1).to(torch.long)

    cu_seqlens = packed_seq_params["cu_seqlens_q"].to(torch.int32)
    max_seqlen = int(packed_seq_params["max_seqlen_q"])
    refiner_cu = refiner_packed_seq_params["cu_seqlens_q"].to(torch.int32)
    refiner_max = int(refiner_packed_seq_params["max_seqlen_q"])

    if x.dim() != 3 or x.shape[0] != 1:
        raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
    seq_len = int(x.shape[1])
    device = x.device

    rope_freqs = dit.rope(img_position_ids).to(device)

    decoder_input, t_emb = dit._embed(
        x=x, audio_x=audio_x, text_embeddings_selected=text_selected,
        unique_timesteps=unique_timesteps.view(-1).to(device),
        img_pos=img_pos.to(device), audio_pos=audio_pos.to(device), text_pos=text_pos.to(device),
        refiner_cu_seqlens=refiner_cu.to(device), refiner_max_seqlen=refiner_max,
        seq_len=seq_len, device=device,
    )

    combined_indices = (inverse_indices * 3 + token_tags.clamp(min=0)).to(device)
    inverse_indices = inverse_indices.to(device)

    cu_seqlens = cu_seqlens.to(device)
    control_hints = controlnet(
        decoder_input,
        control_rows,
        img_pos.to(device),
        audio_pos.to(device),
        t_emb=t_emb,
        combined_indices=combined_indices,
        rope_freqs=rope_freqs,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return control_hints


def model_fn_minimax_h3(
    dit,
    video_latents,
    audio_latents,
    packed,
    prompt_embeds,
    timestep_video,
    timestep_audio,
    keyframe_cond_anchor=None,
    ref_visual_anchor=None,
    ref_audio_anchor=None,
    input_latents_video=None,
    denoise_mask_video=None,
    input_latents_audio=None,
    denoise_mask_audio=None,
    imgvid_cond_noise_aug=0.999,
    audio_cond_noise_aug=1.0,
    controlnet=None,
    control_rows=None,
    control_scale=1.0,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    t_video = 1.0 - float(timestep_video) / 1000
    t_audio = 1.0 - float(timestep_audio) / 1000

    dtype, device = video_latents.dtype, video_latents.device
    f, h, w = video_latents.shape[2:]
    audio_channel, audio_t = audio_latents.shape[0], audio_latents.shape[-1]

    if input_latents_video is not None:
        video_latents = video_latents * denoise_mask_video + input_latents_video * (1.0 - denoise_mask_video)
        denoise_mask_video = patchify_video(denoise_mask_video)[:, 0]
    if input_latents_audio is not None:
        audio_latents = audio_latents * denoise_mask_audio + input_latents_audio * (1.0 - denoise_mask_audio)
        denoise_mask_audio = pack_audio(denoise_mask_audio)[:, 0]
    video_rows = patchify_video(video_latents)
    audio_rows = pack_audio(audio_latents)

    img_pos, audio_pos, text_pos = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
    cu, seq_len, text_len = packed["cu_seqlens"], packed["seq_len"], text_pos.shape[0]
    # Video Sequence
    cond_anchor = ref_visual_anchor if ref_visual_anchor is not None else keyframe_cond_anchor
    cond_rows_count = 0 if cond_anchor is None else cond_anchor.shape[0]
    x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
    x[0].index_copy_(0, img_pos[cond_rows_count:], video_rows)
    if cond_anchor is not None:
        x[0].index_copy_(0, img_pos[:cond_rows_count], cond_anchor)
    # Audio Sequence
    ref_audio_rows_count = 0 if ref_audio_anchor is None else ref_audio_anchor.shape[0]
    audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
    audio_x[0].index_copy_(0, audio_pos[ref_audio_rows_count:], audio_rows)
    if ref_audio_anchor is not None:
        audio_x[0].index_copy_(0, audio_pos[:ref_audio_rows_count], ref_audio_anchor)

    timesteps = torch.full((seq_len,), t_video, dtype=torch.float32, device=device)
    timesteps[audio_pos] = t_audio

    if input_latents_video is not None:
        timesteps[img_pos[cond_rows_count:][denoise_mask_video == 0]] = 1.0
    if input_latents_audio is not None:
        timesteps[audio_pos[ref_audio_rows_count:][denoise_mask_audio == 0]] = 1.0

    timesteps[img_pos[:cond_rows_count]] = max(t_video, imgvid_cond_noise_aug)
    timesteps[audio_pos[:ref_audio_rows_count]] = max(t_audio, audio_cond_noise_aug)
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)

    # ControlNet
    if control_rows is not None:
        control_residual = model_fn_minimax_h3_controlnet(
            dit,
            x,
            audio_x,
            packed["img_position_ids"],
            unique_timesteps,
            inverse_indices,
            packed["token_tags"],
            prompt_embeds,
            {"position_ids": img_pos},
            {"position_ids": audio_pos},
            {"position_ids": text_pos},
            {"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
            {"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            controlnet=controlnet,
            control_rows=control_rows,
        )
        control_hints = {}
        for block_id in range(len(dit.blocks)):
            if block_id in controlnet.control_layers_mapping:
                control_hints[block_id] = control_residual[controlnet.control_layers_mapping[block_id]] * control_scale
    else:
        control_hints = None

    v_video_rows, v_audio_rows = dit(
        x=x,
        audio_x=audio_x,
        img_position_ids=packed["img_position_ids"],
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        token_tags=packed["token_tags"],
        prompt_embeds=prompt_embeds,
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
        refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
        update_mask=None,
        skip_mask_out_condition=True,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        control_hints=control_hints,
    )

    v_video_rows = v_video_rows[cond_rows_count:]
    v_audio_rows = v_audio_rows[ref_audio_rows_count:]

    v_video = unpatchify_video(v_video_rows, f, h, w)
    v_audio = unpack_audio(v_audio_rows, audio_channel, audio_t)
    return -v_video, -v_audio
