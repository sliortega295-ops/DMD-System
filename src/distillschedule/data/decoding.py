"""Actual offline media decoding. Native feature encoding is a separate plugin."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import wave

from .manifest import frame_paths, raw_record, require


def _video(path):
    import numpy as np
    import torch

    path = Path(path)
    if path.is_dir():
        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError("frame directories require the optional Pillow dependency") from exc
        images = []
        for frame in frame_paths(path):
            with Image.open(frame) as image:
                images.append(np.array(image.convert("RGB")))
        require(all(image.shape == images[0].shape for image in images),
                "frame dimensions differ; no implicit resize is allowed")
        return torch.from_numpy(np.stack(images))
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    require(ffmpeg and ffprobe, "video files require existing ffmpeg and ffprobe executables")
    # The manifest hashes this one file, not an external playlist's segments.
    # Only admit self-contained media demuxers and local I/O, in BOTH tools.
    input_options = ["-format_whitelist", "mov,matroska,webm,avi,mpeg,mpegts,flv,ogg,gif",
                     "-protocol_whitelist", "file,pipe"]
    try:
        probe = subprocess.run([ffprobe, "-v", "error", *input_options, "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "json", str(path)],
                               capture_output=True, check=True)
        streams = json.loads(probe.stdout).get("streams", [])
        require(len(streams) == 1, "video requires one selected video stream")
        width, height = streams[0]["width"], streams[0]["height"]
        require(type(width) is int and type(height) is int and width > 0 and height > 0,
                "invalid decoded video geometry")
        decoded = subprocess.run([ffmpeg, "-nostdin", "-v", "error", *input_options, "-noautorotate", "-i", str(path),
                                  "-map", "0:v:0", "-an", "-sn", "-vsync", "0", "-threads", "1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                 capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode video: {path}") from exc
    frame_bytes = width * height * 3
    require(bool(decoded) and len(decoded) % frame_bytes == 0, "decoded video has incomplete frames")
    # This is a byte-layout decode, not a latent-space resize or frame resample.
    return torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy().reshape(-1, height, width, 3))


def _audio(path):
    import numpy as np
    import torch

    try:
        with wave.open(str(path), "rb") as source:
            channels, rate, width = source.getnchannels(), source.getframerate(), source.getsampwidth()
            count = source.getnframes()
            require(source.getcomptype() == "NONE" and width in (1, 2, 3, 4),
                    "audio decoder requires uncompressed 8/16/24/32-bit PCM WAV")
            data = source.readframes(count)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"cannot decode PCM WAV: {path}") from exc
    require(count > 0 and rate > 0 and channels > 0 and len(data) == count * channels * width,
            "WAV has no samples or an incomplete PCM payload")
    if width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 3:
        octets = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        integers = (integers ^ 0x800000) - 0x800000
        samples = integers.astype(np.float32) / 8388608
    else:
        samples = np.frombuffer(data, dtype="<i2" if width == 2 else "<i4").astype(np.float32)
        samples /= 32768 if width == 2 else 2147483648
    return {"samples": torch.from_numpy(samples.reshape(count, channels).copy()), "sample_rate": rate}


def decode_record(record: dict, *, base_dir: str | Path | None = None) -> dict:
    """Decode one explicit raw record, preserving id and metadata.

    Video is THWC RGB uint8. Audio is {samples: TC float32, sample_rate: int}.
    Text is unchanged UTF-8. No trimming, padding, resizing or resampling occurs.
    """
    resolved, _ = raw_record(record, Path.cwd() if base_dir is None else Path(base_dir))
    result = deepcopy(resolved)
    if "video" in resolved:
        result["video"] = _video(resolved["video"])
    if "audio" in resolved:
        result["audio"] = _audio(resolved["audio"])
    if "text" in resolved:
        reference = resolved["text"]
        result["text"] = (Path(reference["path"]).read_text(encoding="utf-8") if "path" in reference
                          else reference["value"])
    return result
