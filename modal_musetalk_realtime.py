"""Test 2: persistent, low-latency MuseTalk realtime worker.

This is a separate Modal worker from the batch `MuseTalkWorker` in
`modal_musetalk.py`. The existing batch endpoint and scripts/inference.py
fallback stay untouched.

Run:
    modal run --env main modal_musetalk_realtime.py::test_realtime
    modal run --env main modal_musetalk_realtime.py::test_realtime_short
"""

import time

import modal

from modal_musetalk import (
    APP_NAME,
    AVATARS_DIR,
    MODELS_DIR,
    OUTPUT_SUBDIR,
    REPO_DIR,
    app,
    avatars,
    musetalk_image,
)
from modal_musetalk import models as models_volume

# `modal_musetalk_realtime.py` imports the sibling `modal_musetalk` module at
# top level (for APP_NAME, volumes, musetalk_image, etc.). That import must
# succeed again when a container reconstructs this file remotely, so the
# module needs to be bundled into the image explicitly. `musetalk_image`
# itself is left untouched (add_local_python_source returns a new Image),
# so the existing batch MuseTalkWorker/image is unaffected.
#
# modal_musetalk.py only imports the stdlib `modal` package at module level,
# so no other local project modules need to be bundled here.
realtime_image = musetalk_image.add_local_python_source("modal_musetalk")

REPO_MODELS_LINK = f"{REPO_DIR}/models"

DEFAULT_VERSION = "v15"
DEFAULT_AVATAR_ID = "saudi-female"
DEFAULT_AVATAR_VIDEO = "source/saudi-female.mp4"
DEFAULT_BBOX_SHIFT = 0
DEFAULT_EXTRA_MARGIN = 10
DEFAULT_PARSING_MODE = "jaw"
DEFAULT_LEFT_CHEEK_WIDTH = 90
DEFAULT_RIGHT_CHEEK_WIDTH = 90
DEFAULT_FPS = 25
DEFAULT_BATCH_SIZE = 8
REALTIME_CACHE_SUBDIR = "realtime_cache"


def _ensure_models_symlink():
    import os
    import shutil

    if os.path.islink(REPO_MODELS_LINK):
        if os.readlink(REPO_MODELS_LINK) == MODELS_DIR:
            return
        os.unlink(REPO_MODELS_LINK)
    elif os.path.isdir(REPO_MODELS_LINK):
        shutil.rmtree(REPO_MODELS_LINK)
    elif os.path.exists(REPO_MODELS_LINK):
        os.remove(REPO_MODELS_LINK)

    os.symlink(MODELS_DIR, REPO_MODELS_LINK)
    print(f"linked {REPO_MODELS_LINK} -> {MODELS_DIR}")


def _load_model_bundle(device: str, version: str = DEFAULT_VERSION, use_float16: bool = True) -> dict:
    import os

    from transformers import WhisperModel

    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import load_all_model

    if version == "v15":
        unet_model_path = f"{MODELS_DIR}/musetalkV15/unet.pth"
        unet_config = f"{MODELS_DIR}/musetalkV15/musetalk.json"
    elif version == "v1":
        unet_model_path = f"{MODELS_DIR}/musetalk/pytorch_model.bin"
        unet_config = f"{MODELS_DIR}/musetalk/musetalk.json"
    else:
        raise ValueError(f"version must be 'v15' or 'v1', got {version!r}")

    for label, path in (("unet weights", unet_model_path), ("unet config", unet_config)):
        if not os.path.isfile(path):
            raise RuntimeError(f"{version} {label} missing: {path}")

    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type="sd-vae",
        unet_config=unet_config,
        device=device,
    )

    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    weight_dtype = unet.model.dtype
    whisper_dir = f"{MODELS_DIR}/whisper"
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    return {
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "whisper": whisper,
        "audio_processor": audio_processor,
        "weight_dtype": weight_dtype,
    }


def _load_face_parser(left_cheek_width: int, right_cheek_width: int):
    from musetalk.utils.face_parsing import FaceParsing

    return FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)


# Cache layout version. Bump this whenever the on-disk artifact format
# changes so stale caches (e.g. from an older, slower layout) are
# transparently recomputed instead of misread.
CACHE_LAYOUT_VERSION = 2


def _avatar_cache_paths(avatar_id: str, version: str) -> dict:
    import os

    base = os.path.join(AVATARS_DIR, REALTIME_CACHE_SUBDIR, version, avatar_id)
    return {
        "base": base,
        # Each of these is a single consolidated file (list of numpy arrays /
        # tensors) rather than one-file-per-frame. calm-avatar-assets is a
        # networked Modal Volume, so hundreds of small per-frame reads incur
        # a per-file round trip and are far slower than a handful of large
        # sequential reads, even though the total bytes are similar.
        "full_imgs": os.path.join(base, "full_imgs.pkl"),
        "coords": os.path.join(base, "coords.pkl"),
        "latents": os.path.join(base, "latents.pt"),
        "mask": os.path.join(base, "mask.pkl"),
        "mask_coords": os.path.join(base, "mask_coords.pkl"),
        "info": os.path.join(base, "avatar_info.json"),
    }


def _extract_frames_from_video(video_path: str, work_dir: str) -> list:
    import glob
    import os

    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.system(f"ffmpeg -v fatal -i {video_path} -start_number 0 {frames_dir}/%08d.png")
    return sorted(glob.glob(os.path.join(frames_dir, "*.[jpJP][pnPN]*[gG]")))


def _pipe_frames_to_ffmpeg(composited_frames, video_fps, source_audio, destination, encode_preset=None):
    """Encode already-composited, in-RAM BGR frames + mux `source_audio`
    into `destination` with a single ffmpeg process (raw video on stdin,
    audio as a second input) -- no intermediate PNGs or silent .mp4.

    `encode_preset`, when given, is passed through as ffmpeg's `-preset`
    (e.g. "ultrafast"); omitted, ffmpeg uses its own default ("medium").
    Resolution, fps, and pixel format are unaffected by the preset choice.

    Returns the wall-clock seconds spent encoding+muxing.
    """
    import subprocess
    import time as _time

    if not composited_frames:
        raise RuntimeError("compositing produced no frames")

    start = _time.time()
    frame_h, frame_w = composited_frames[0].shape[:2]
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}", "-r", str(video_fps),
        "-i", "pipe:0",
        "-i", source_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264",
    ]
    if encode_preset:
        ffmpeg_cmd += ["-preset", encode_preset]
    ffmpeg_cmd += [
        "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest",
        destination,
    ]

    # Build the raw frame buffer once and hand it to communicate() in a
    # single call. subprocess.communicate() owns writing to stdin (and
    # closing it) internally using a helper thread when stderr/stdout are
    # also piped; writing incrementally and then closing stdin ourselves
    # races with that and raises "ValueError: flush of closed file".
    raw_video_bytes = b"".join(frame.tobytes() for frame in composited_frames)
    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, stderr = proc.communicate(input=raw_video_bytes)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg piped encode/mux failed (returncode={proc.returncode}): "
            f"{stderr.decode(errors='ignore')}"
        )
    return _time.time() - start


def _compute_alpha_list(coord_list, mask_list, mask_coords_list):
    """Precompute the per-frame blend alpha used by the NumPy/OpenCV
    compositor (see infer()'s "numpy" pipeline_mode).

    `get_image_blending` (musetalk/utils/blending.py) does, per frame:
      1. crop the full frame to `crop_box` (the larger, precomputed mask
         region) -> `face_large`
      2. overwrite `face_large` at `face_box` (the tighter face bbox) with
         the raw generated face pixels (a plain, unmasked paste)
      3. alpha-blend `face_large` back over the ORIGINAL full frame within
         `crop_box`, using `mask_array` (already `crop_box`-sized) as the
         per-pixel alpha
    Outside `face_box` but inside `crop_box`, step 2 leaves `face_large`
    identical to the original frame, so step 3's blend there is
    mathematically a no-op (blending a value with itself reproduces that
    same value regardless of alpha). The only region whose output value
    actually depends on the generated face is `face_box` itself, blended
    against the *sub-region* of `mask_array` that overlaps `face_box`.

    This function precomputes exactly that sub-region -- `mask_array`
    cropped from crop_box-relative to face_box-relative coordinates -- as a
    float32 alpha in [0, 1] with shape (h, w, 1), once per avatar frame
    (avatar-frame-dependent only), so the per-request hot path never repeats
    this slicing/dtype conversion.
    """
    import numpy as np

    alpha_list = []
    for bbox, mask, crop_box in zip(coord_list, mask_list, mask_coords_list):
        x1, y1, x2, y2 = bbox
        x_s, y_s, _x_e, _y_e = crop_box
        sub_mask = mask[y1 - y_s : y2 - y_s, x1 - x_s : x2 - x_s]
        alpha = np.ascontiguousarray(sub_mask, dtype=np.float32) / 255.0
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]
        alpha_list.append(alpha)
    return alpha_list


def _prepare_avatar(
    *,
    avatar_id: str,
    video_path_abs: str,
    video_path_rel: str,
    version: str,
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    fps: int,
    vae,
    face_parser,
    force_recreate: bool = False,
):
    import json
    import os
    import pickle
    import shutil
    import time as _time

    import torch

    from musetalk.utils.blending import get_image_prepare_material
    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
    from musetalk.utils.utils import get_video_fps

    paths = _avatar_cache_paths(avatar_id, version)
    avatar_info = {
        "avatar_id": avatar_id,
        "video_path": video_path_rel,
        "bbox_shift": bbox_shift,
        "version": version,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "cache_layout": CACHE_LAYOUT_VERSION,
    }

    cached = False
    existing = {}
    if not force_recreate and os.path.isfile(paths["info"]):
        try:
            with open(paths["info"]) as f:
                existing = json.load(f)
            cached = all(existing.get(k) == v for k, v in avatar_info.items())
        except Exception:
            cached = False

    t0 = _time.time()
    frame_list_cycle = None
    coord_list_cycle = None
    input_latent_list_cycle = None
    mask_list_cycle = None
    mask_coords_list_cycle = None
    alpha_list_cycle = None
    video_fps = fps

    if cached:
        try:
            device_for_load = "cuda" if torch.cuda.is_available() else "cpu"
            # Each of these is one file read (not hundreds), so a warm cache
            # hit is fast regardless of how many frames the avatar has.
            input_latent_list = torch.load(paths["latents"], map_location=device_for_load)
            with open(paths["coords"], "rb") as f:
                coord_list = pickle.load(f)
            with open(paths["full_imgs"], "rb") as f:
                frame_list = pickle.load(f)
            with open(paths["mask_coords"], "rb") as f:
                mask_coords_list = pickle.load(f)
            with open(paths["mask"], "rb") as f:
                mask_list = pickle.load(f)
            video_fps = existing.get("video_fps", fps)
            if not frame_list or not mask_list or not coord_list or not input_latent_list:
                raise RuntimeError("cached avatar assets are incomplete")

            # Alpha for the NumPy/OpenCV compositor is derived purely from
            # already-cached mask/coord data (cheap slicing, no
            # face-parsing), so it is (re)computed here rather than also
            # being persisted as its own cache file.
            alpha_list = _compute_alpha_list(coord_list, mask_list, mask_coords_list)

            # Reconstruct the "ping-pong" playback cycle in memory (cheap
            # list concatenation) instead of persisting/reading it twice.
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            mask_list_cycle = mask_list + mask_list[::-1]
            mask_coords_list_cycle = mask_coords_list + mask_coords_list[::-1]
            alpha_list_cycle = alpha_list + alpha_list[::-1]
        except Exception as exc:
            print(f"avatar cache for {avatar_id!r} unusable ({exc}); recomputing")
            cached = False

    if not cached:
        import cv2

        if os.path.isdir(paths["base"]):
            shutil.rmtree(paths["base"])
        os.makedirs(paths["base"], exist_ok=True)

        if not os.path.isfile(video_path_abs):
            raise FileNotFoundError(f"avatar source video not found: {video_path_abs}")

        video_fps = get_video_fps(video_path_abs) or fps
        work_dir = f"/tmp/musetalk_realtime_prep/{avatar_id}"
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        img_list = _extract_frames_from_video(video_path_abs, work_dir)
        if not img_list:
            raise RuntimeError(f"no frames extracted from {video_path_abs}")

        print(f"extracting landmarks for avatar {avatar_id!r} ({len(img_list)} frames)...")
        coord_list, frame_list = get_landmark_and_bbox(img_list, bbox_shift)

        input_latent_list = []
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if version == "v15":
                y2 = min(y2 + extra_margin, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            input_latent_list.append(vae.get_latents_for_unet(crop_frame))

        # Face-parsing masks are computed once per unique source frame
        # (not once per cycle position) since the "ping-pong" second half
        # revisits the same frames in reverse order. Halves face-parsing
        # work compared to iterating the full cycle.
        mask_coords_list = []
        mask_list = []
        for idx, frame in enumerate(frame_list):
            x1, y1, x2, y2 = coord_list[idx]
            mode = parsing_mode if version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], fp=face_parser, mode=mode
            )
            mask_coords_list.append(crop_box)
            mask_list.append(mask)

        # Persist only the non-cycled (single-pass) artifacts; the
        # "ping-pong" cycle is reconstructed in memory on every load (here
        # and on cache hits) via cheap list concatenation.
        with open(paths["full_imgs"], "wb") as f:
            pickle.dump(frame_list, f)
        with open(paths["mask"], "wb") as f:
            pickle.dump(mask_list, f)
        with open(paths["mask_coords"], "wb") as f:
            pickle.dump(mask_coords_list, f)
        with open(paths["coords"], "wb") as f:
            pickle.dump(coord_list, f)
        torch.save(input_latent_list, paths["latents"])

        avatar_info["video_fps"] = video_fps
        with open(paths["info"], "w") as f:
            json.dump(avatar_info, f)

        avatars.commit()
        shutil.rmtree(work_dir, ignore_errors=True)

        alpha_list = _compute_alpha_list(coord_list, mask_list, mask_coords_list)

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        mask_list_cycle = mask_list + mask_list[::-1]
        mask_coords_list_cycle = mask_coords_list + mask_coords_list[::-1]
        alpha_list_cycle = alpha_list + alpha_list[::-1]

    elapsed = _time.time() - t0
    return {
        "frame_list_cycle": frame_list_cycle,
        "coord_list_cycle": coord_list_cycle,
        "input_latent_list_cycle": input_latent_list_cycle,
        "mask_list_cycle": mask_list_cycle,
        "mask_coords_list_cycle": mask_coords_list_cycle,
        "alpha_list_cycle": alpha_list_cycle,
        "video_fps": video_fps,
    }, elapsed, ("cache" if cached else "computed")


@app.cls(
    image=realtime_image,
    gpu="L4",
    volumes={MODELS_DIR: models_volume, AVATARS_DIR: avatars},
    timeout=3600,
    scaledown_window=300,
)
class MuseTalkRealtimeWorker:
    @modal.enter()
    def start_container(self):
        import os
        import time as _time
        import traceback
        import uuid

        import torch

        enter_started = _time.time()
        # A fresh ID per container lifecycle (not per request), so a
        # benchmark can prove two requests landed on the same warm
        # container instead of two independently cold-started ones.
        self.container_id = uuid.uuid4().hex[:12]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = None
        self.face_parser = None
        self.avatar_cache = {}
        self._served_requests = 0
        self._model_loading_seconds = 0.0
        self._avatar_preparation_seconds = 0.0
        self._avatar_prep_source = None
        self._startup_error = None

        if not os.path.isdir(REPO_DIR):
            raise RuntimeError("MuseTalk repository is missing from the container image")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the GPU container")

        _ensure_models_symlink()
        os.chdir(REPO_DIR)

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")

        try:
            model_start = _time.time()
            self.models = _load_model_bundle(device=self.device, version=DEFAULT_VERSION, use_float16=True)
            self.face_parser = _load_face_parser(DEFAULT_LEFT_CHEEK_WIDTH, DEFAULT_RIGHT_CHEEK_WIDTH)
            self._model_loading_seconds = _time.time() - model_start
            print(f"models loaded in {self._model_loading_seconds:.2f}s")

            avatar_start = _time.time()
            avatar_data, _, source = _prepare_avatar(
                avatar_id=DEFAULT_AVATAR_ID,
                video_path_abs=os.path.join(AVATARS_DIR, DEFAULT_AVATAR_VIDEO),
                video_path_rel=DEFAULT_AVATAR_VIDEO,
                version=DEFAULT_VERSION,
                bbox_shift=DEFAULT_BBOX_SHIFT,
                extra_margin=DEFAULT_EXTRA_MARGIN,
                parsing_mode=DEFAULT_PARSING_MODE,
                fps=DEFAULT_FPS,
                vae=self.models["vae"],
                face_parser=self.face_parser,
            )
            self.avatar_cache[DEFAULT_AVATAR_ID] = avatar_data
            self._avatar_preparation_seconds = _time.time() - avatar_start
            self._avatar_prep_source = source
            print(
                f"avatar {DEFAULT_AVATAR_ID!r} ready in "
                f"{self._avatar_preparation_seconds:.2f}s (source={source})"
            )
        except Exception as exc:
            traceback.print_exc()
            self._startup_error = f"{type(exc).__name__}: {exc}"

        self._enter_seconds = _time.time() - enter_started
        print(f"container_id={self.container_id} ready in {self._enter_seconds:.2f}s")

    @modal.method()
    def infer(
        self,
        audio_path: str = "audio/test-arabic.wav",
        avatar_id: str = DEFAULT_AVATAR_ID,
        output_name: str = None,
        fps: int = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pipeline_mode: str = "numpy",
        encode_preset: str = None,
    ):
        """Run one realtime inference request against an already-warm avatar.

        pipeline_mode:
          "numpy" (default) -- same as "optimized" (in-RAM frames, piped
              ffmpeg) but replaces the PIL-based get_image_blending call
              with a direct NumPy/OpenCV blend using an alpha mask
              precomputed once per avatar frame in _prepare_avatar (see
              _compute_alpha_list). No Image.fromarray/crop/paste/np.array
              round trip; the blend math is unchanged.
          "optimized" -- no per-frame deepcopy of the static avatar
              background (safe: get_image_blending never mutates its
              `image` argument, see comment below), composited frames are
              kept in RAM and piped directly into a single ffmpeg process
              that also muxes the audio, so nothing is written to disk and
              read back for encoding. Uses the original PIL-based
              get_image_blending. Kept so the NumPy compositor can be
              benchmarked against a like-for-like PIL "before" run.
          "baseline" -- the original approach this replaces: per-frame
              copy.deepcopy + cv2.imwrite of a PNG to /tmp, then a separate
              ffmpeg image2 pass to encode video, then a second ffmpeg pass
              to mux audio. Kept only so later optimizations can be
              benchmarked against a like-for-like "before" run on the same
              warm container.

          None of the three modes changes the MuseTalk model, UNet/VAE
          inference, the face-parsing/blending algorithm, or image
          quality -- all three reproduce the same blend result ("optimized"
          and "baseline" call the exact same get_image_blending with the
          exact same cached masks/coords; "numpy" reproduces the same
          blending mathematics directly, see _compute_alpha_list's
          docstring for the derivation).

        encode_preset: optional ffmpeg `-preset` value (e.g. "ultrafast")
            for the piped "numpy"/"optimized" encoder. None uses ffmpeg's
            default ("medium"), matching prior behavior. Resolution and fps
            are unaffected either way.
        """
        import copy
        import os
        import shutil
        import subprocess
        import time as _time
        import traceback
        import uuid

        import cv2
        import numpy as np
        import torch

        from musetalk.utils.blending import get_image_blending
        from musetalk.utils.utils import datagen

        request_start = _time.time()
        container_cold_start = self._served_requests == 0
        self._served_requests += 1

        peak_gpu_memory_mb = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            if self._startup_error:
                raise RuntimeError(f"container warm-up failed: {self._startup_error}")
            if self.models is None:
                raise RuntimeError("models are not loaded on this worker")
            if pipeline_mode not in ("optimized", "baseline", "numpy"):
                raise ValueError(
                    f"pipeline_mode must be 'numpy', 'optimized' or 'baseline', got {pipeline_mode!r}"
                )

            avatars.reload()
            source_audio = os.path.join(AVATARS_DIR, audio_path)
            if not os.path.isfile(source_audio):
                raise FileNotFoundError(f"audio not found in calm-avatar-assets volume: {source_audio}")

            avatar_data = self.avatar_cache.get(avatar_id)
            if avatar_data is None:
                raise ValueError(f"avatar {avatar_id!r} is not prepared on this worker")

            device = self.device
            vae = self.models["vae"]
            unet = self.models["unet"]
            pe = self.models["pe"]
            whisper = self.models["whisper"]
            audio_processor = self.models["audio_processor"]
            weight_dtype = self.models["weight_dtype"]
            timesteps = torch.tensor([0], device=device)

            # Everything below is read-only avatar-specific state prepared
            # once in @modal.enter() (see _prepare_avatar): face coords,
            # masks, mask coords, and the static background frames are never
            # recomputed per request.
            frame_list_cycle = avatar_data["frame_list_cycle"]
            coord_list_cycle = avatar_data["coord_list_cycle"]
            input_latent_list_cycle = avatar_data["input_latent_list_cycle"]
            mask_list_cycle = avatar_data["mask_list_cycle"]
            mask_coords_list_cycle = avatar_data["mask_coords_list_cycle"]
            alpha_list_cycle = avatar_data.get("alpha_list_cycle")
            if pipeline_mode == "numpy" and not alpha_list_cycle:
                raise RuntimeError(
                    "avatar cache has no precomputed alpha (re-run avatar preparation)"
                )
            video_fps = fps or avatar_data.get("video_fps") or DEFAULT_FPS

            audio_start = _time.time()
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(source_audio)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=video_fps,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )
            audio_feature_extraction_seconds = _time.time() - audio_start
            audio_duration_seconds = librosa_length / 16000.0

            # UNet forward pass ("MuseTalk inference"), VAE decode, and
            # frame collection are timed separately. VAE decode is already
            # batched here -- `vae.decode_latents()` is called once per
            # `datagen` batch (size=`batch_size`) on a stacked latent
            # tensor, not once per individual frame.
            unet_inference_seconds = 0.0
            vae_decode_seconds = 0.0
            frame_collection_seconds = 0.0
            res_frame_list = []
            sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
            with torch.no_grad():
                for whisper_batch, latent_batch in datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list_cycle,
                    batch_size=batch_size,
                    delay_frame=0,
                    device=device,
                ):
                    unet_start = _time.time()
                    audio_feature_batch = pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=unet.model.dtype)
                    pred_latents = unet.model(
                        latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
                    ).sample
                    sync()
                    unet_inference_seconds += _time.time() - unet_start

                    vae_start = _time.time()
                    recon = vae.decode_latents(pred_latents)  # batched GPU decode + one CPU transfer
                    sync()
                    vae_decode_seconds += _time.time() - vae_start

                    collect_start = _time.time()
                    for res_frame in recon:
                        res_frame_list.append(res_frame)
                    frame_collection_seconds += _time.time() - collect_start

            num_output_frames = len(res_frame_list)

            stem = output_name or f"{avatar_id}_{os.path.splitext(os.path.basename(audio_path))[0]}_realtime"
            if stem.endswith(".mp4"):
                stem = stem[:-4]
            output_vid_name = f"{stem}.mp4"
            output_rel_path = os.path.join(OUTPUT_SUBDIR, output_vid_name)
            destination = os.path.join(AVATARS_DIR, output_rel_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            compositing_seconds = 0.0
            imwrite_seconds = 0.0
            work_dir = None

            if pipeline_mode == "baseline":
                run_id = uuid.uuid4().hex[:8]
                work_dir = f"/tmp/musetalk_realtime/{run_id}"
                frames_out_dir = os.path.join(work_dir, "frames")
                os.makedirs(frames_out_dir, exist_ok=True)

                for i, res_frame in enumerate(res_frame_list):
                    bbox = coord_list_cycle[i % len(coord_list_cycle)]
                    comp_start = _time.time()
                    # Unnecessary in practice (get_image_blending never
                    # mutates its `image` argument -- see the "optimized"
                    # branch below) but kept here so "baseline" reproduces
                    # the exact original per-frame cost being optimized away.
                    ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                    x1, y1, x2, y2 = bbox
                    try:
                        res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception:
                        continue
                    mask = mask_list_cycle[i % len(mask_list_cycle)]
                    mask_crop_box = mask_coords_list_cycle[i % len(mask_coords_list_cycle)]
                    combined = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
                    compositing_seconds += _time.time() - comp_start

                    imwrite_start = _time.time()
                    cv2.imwrite(os.path.join(frames_out_dir, f"{i:08d}.png"), combined)
                    imwrite_seconds += _time.time() - imwrite_start

                encoding_start = _time.time()
                temp_vid_path = os.path.join(work_dir, f"temp_{stem}.mp4")
                os.system(
                    f"ffmpeg -y -v warning -r {video_fps} -f image2 "
                    f"-i {frames_out_dir}/%08d.png -vcodec libx264 -vf format=yuv420p "
                    f"-crf 18 {temp_vid_path}"
                )
                mux = subprocess.run(
                    f"ffmpeg -y -v warning -i {source_audio} -i {temp_vid_path} {destination}",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if mux.returncode != 0:
                    raise RuntimeError(f"ffmpeg audio/video mux failed: {mux.stderr}")
                encoding_seconds = _time.time() - encoding_start
            elif pipeline_mode == "optimized":
                # "optimized": no disk I/O in the hot path. Composited
                # frames are kept in RAM and piped straight into a single
                # ffmpeg process (raw video on stdin + the audio file as a
                # second input) that encodes AND muxes in one pass -- no
                # intermediate PNGs, no intermediate silent .mp4. Still uses
                # the original PIL-based get_image_blending (see "numpy"
                # below for the direct NumPy/OpenCV replacement).
                composited_frames = []
                for i, res_frame in enumerate(res_frame_list):
                    bbox = coord_list_cycle[i % len(coord_list_cycle)]
                    comp_start = _time.time()
                    x1, y1, x2, y2 = bbox
                    try:
                        res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception:
                        continue
                    mask = mask_list_cycle[i % len(mask_list_cycle)]
                    mask_crop_box = mask_coords_list_cycle[i % len(mask_coords_list_cycle)]
                    # get_image_blending's first step is
                    # `Image.fromarray(image[:, :, ::-1])`. Because the
                    # channel-reversing slice has a negative stride, Pillow
                    # cannot alias that buffer and copies it internally
                    # (`Image.fromarray` falls back to `obj.tobytes()`
                    # whenever `strides` is set). So `image` is never
                    # mutated here, and the per-frame `copy.deepcopy` of the
                    # static avatar frame in the baseline path was always
                    # redundant -- the static background frame can be
                    # passed directly.
                    combined = get_image_blending(
                        frame_list_cycle[i % len(frame_list_cycle)],
                        res_frame_resized,
                        bbox,
                        mask,
                        mask_crop_box,
                    )
                    composited_frames.append(np.ascontiguousarray(combined, dtype=np.uint8))
                    compositing_seconds += _time.time() - comp_start

                encoding_start = _time.time()
                encoding_seconds = _pipe_frames_to_ffmpeg(
                    composited_frames, video_fps, source_audio, destination, encode_preset
                )
            else:
                # "numpy": same in-RAM/piped-ffmpeg approach as "optimized",
                # but the PIL Image.fromarray/crop/paste/np.array round trip
                # in get_image_blending is replaced with a direct
                # NumPy/OpenCV blend restricted to the face bbox, using the
                # alpha precomputed once per avatar frame during avatar
                # preparation (_compute_alpha_list). See that function's
                # docstring for why this reproduces the exact same pixels
                # as get_image_blending without needing to touch anything
                # outside the (smaller) face bbox region.
                composited_frames = []
                for i, res_frame in enumerate(res_frame_list):
                    idx = i % len(coord_list_cycle)
                    bbox = coord_list_cycle[idx]
                    comp_start = _time.time()
                    x1, y1, x2, y2 = bbox
                    try:
                        res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception:
                        continue
                    alpha = alpha_list_cycle[idx % len(alpha_list_cycle)]
                    background = frame_list_cycle[idx % len(frame_list_cycle)]

                    out = background.copy()  # never mutate the shared cached avatar frame
                    roi = out[y1:y2, x1:x2].astype(np.float32)
                    face_f = res_frame_resized.astype(np.float32)
                    blended = face_f * alpha + roi * (1.0 - alpha)
                    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

                    composited_frames.append(out)
                    compositing_seconds += _time.time() - comp_start

                encoding_start = _time.time()
                encoding_seconds = _pipe_frames_to_ffmpeg(
                    composited_frames, video_fps, source_audio, destination, encode_preset
                )

            if not os.path.isfile(destination):
                raise RuntimeError("Inference produced no output video")

            size_bytes = os.path.getsize(destination)
            avatars.commit()
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

            # Excludes @modal.enter() startup entirely: request_start is
            # captured at the top of this method, which only runs after the
            # container (and its models/avatar) are already warm.
            total_request_seconds = _time.time() - request_start
            vae_decode_compositing_seconds = vae_decode_seconds + compositing_seconds
            processing_seconds = (
                audio_feature_extraction_seconds
                + unet_inference_seconds
                + vae_decode_compositing_seconds
                + encoding_seconds
            )
            realtime_factor = (
                total_request_seconds / audio_duration_seconds if audio_duration_seconds > 0 else None
            )
            effective_generated_fps = (
                num_output_frames / (unet_inference_seconds + vae_decode_compositing_seconds)
                if (unet_inference_seconds + vae_decode_compositing_seconds) > 0
                else None
            )
            if torch.cuda.is_available():
                peak_gpu_memory_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

            return {
                "status": "ok",
                "app": APP_NAME,
                "worker": "realtime",
                "container_id": self.container_id,
                "pipeline_mode": pipeline_mode,
                "encode_preset": encode_preset or "default",
                "batch_size": batch_size,
                "fps_used": video_fps,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "avatar_id": avatar_id,
                "audio_path": audio_path,
                "output_path": output_rel_path,
                "output_volume": "calm-avatar-assets",
                "output_size_mb": round(size_bytes / 1e6, 2),
                "container_cold_start": container_cold_start,
                "cold_container_startup_seconds": round(self._enter_seconds, 3),
                "model_loading_seconds": round(self._model_loading_seconds, 3),
                "avatar_preparation_seconds": round(self._avatar_preparation_seconds, 3),
                "avatar_preparation_source": self._avatar_prep_source,
                "audio_duration_seconds": round(audio_duration_seconds, 3),
                "audio_feature_extraction_seconds": round(audio_feature_extraction_seconds, 3),
                "unet_inference_seconds": round(unet_inference_seconds, 3),
                "vae_decode_seconds": round(vae_decode_seconds, 3),
                "frame_collection_seconds": round(frame_collection_seconds, 5),
                "compositing_seconds": round(compositing_seconds, 3),
                "imwrite_seconds": round(imwrite_seconds, 3),
                "vae_decode_compositing_seconds": round(vae_decode_compositing_seconds, 3),
                "video_encoding_muxing_seconds": round(encoding_seconds, 3),
                "processing_seconds": round(processing_seconds, 3),
                "total_request_seconds": round(total_request_seconds, 3),
                "realtime_factor": round(realtime_factor, 3) if realtime_factor is not None else None,
                "num_output_frames": num_output_frames,
                "effective_generated_fps": (
                    round(effective_generated_fps, 2) if effective_generated_fps is not None else None
                ),
            }
        except Exception as exc:
            traceback.print_exc()
            if torch.cuda.is_available():
                peak_gpu_memory_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            return {
                "status": "error",
                "app": APP_NAME,
                "worker": "realtime",
                "container_id": getattr(self, "container_id", None),
                "pipeline_mode": pipeline_mode,
                "encode_preset": encode_preset or "default",
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "total_request_seconds": round(_time.time() - request_start, 3),
            }

    @modal.method()
    def status(self):
        import traceback

        try:
            import torch

            return {
                "status": "error" if self._startup_error else "ready",
                "app": APP_NAME,
                "worker": "realtime",
                "container_id": self.container_id,
                "device": str(self.device),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "models_loaded": self.models is not None,
                "avatars_resident": list(self.avatar_cache),
                "cold_container_startup_seconds": round(self._enter_seconds, 3),
                "model_loading_seconds": round(self._model_loading_seconds, 3),
                "avatar_preparation_seconds": round(self._avatar_preparation_seconds, 3),
                "avatar_preparation_source": self._avatar_prep_source,
                "served_requests": self._served_requests,
                "startup_error": self._startup_error,
            }
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "app": APP_NAME, "error_type": type(exc).__name__, "error_message": str(exc)}


def _run_realtime_test(audio: str, avatar_id: str, output: str, runs: int):
    worker = MuseTalkRealtimeWorker()
    last_result = None

    for i in range(runs):
        client_start = time.time()
        result = worker.infer.remote(audio_path=audio, avatar_id=avatar_id, output_name=output)
        client_elapsed = time.time() - client_start
        last_result = result

        print(f"\n--- Request {i + 1}/{runs} ({'cold' if i == 0 else 'warm'} expected) ---")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print(f"  client_observed_seconds: {round(client_elapsed, 3)}")

        if result.get("status") != "ok":
            raise SystemExit("Realtime inference failed. Check the error above and Modal logs.")

    print("\nTest completed successfully.")
    print("Download the generated MP4 with:")
    print(
        "  modal volume get --env main calm-avatar-assets "
        f'"{last_result["output_path"]}" .'
    )


@app.local_entrypoint()
def test_realtime(
    audio: str = "audio/test-arabic.wav",
    avatar_id: str = DEFAULT_AVATAR_ID,
    output: str = "saudi-female-arabic-test-realtime.mp4",
    runs: int = 2,
):
    print("\nMuseTalk realtime worker test (full clip)")
    print("------------------------------------------")
    _run_realtime_test(audio=audio, avatar_id=avatar_id, output=output, runs=runs)


@app.local_entrypoint()
def test_realtime_short(
    audio: str = "audio/test-arabic-short.wav",
    avatar_id: str = DEFAULT_AVATAR_ID,
    output: str = "saudi-female-arabic-short-realtime.mp4",
    runs: int = 2,
):
    print("\nMuseTalk realtime worker test (short clip, conversational latency)")
    print("--------------------------------------------------------------------")
    _run_realtime_test(audio=audio, avatar_id=avatar_id, output=output, runs=runs)


@app.local_entrypoint()
def realtime_status():
    print("Checking MuseTalk realtime worker status...")
    result = MuseTalkRealtimeWorker().status.remote()
    for key, value in result.items():
        print(f"  {key}: {value}")


_BENCHMARK_FIELD_ORDER = [
    ("container_id", "container ID"),
    ("pipeline_mode", "pipeline mode"),
    ("encode_preset", "ffmpeg encode preset"),
    ("batch_size", "batch size"),
    ("fps_used", "fps used"),
    ("peak_gpu_memory_mb", "peak GPU memory (MB)"),
    ("container_cold_start", "container cold start (this request)"),
    ("cold_container_startup_seconds", "@modal.enter() startup seconds (model load + avatar prep)"),
    ("model_loading_seconds", "  of which: model loading seconds"),
    ("avatar_preparation_seconds", "  of which: avatar preparation seconds"),
    ("avatar_preparation_source", "  avatar preparation source"),
    ("audio_duration_seconds", "audio duration seconds"),
    ("audio_feature_extraction_seconds", "audio feature extraction seconds"),
    ("unet_inference_seconds", "UNet/MuseTalk inference seconds"),
    ("vae_decode_seconds", "  VAE decode GPU time (batched, seconds)"),
    ("frame_collection_seconds", "  frame collection time (GPU->list, seconds)"),
    ("compositing_seconds", "  face blending/compositing time (seconds)"),
    ("imwrite_seconds", "  cv2.imwrite time (0 in optimized mode, seconds)"),
    ("vae_decode_compositing_seconds", "VAE decode + compositing combined (seconds)"),
    ("video_encoding_muxing_seconds", "video encoding/muxing seconds"),
    ("total_request_seconds", "total inference request seconds (excludes @modal.enter)"),
    ("realtime_factor", "realtime factor (total_request_seconds / audio_duration_seconds)"),
    ("num_output_frames", "number of output frames"),
    ("effective_generated_fps", "effective generated FPS"),
    ("output_path", "output path (in calm-avatar-assets)"),
]


def _print_benchmark_result(label: str, result: dict, client_observed_seconds: float):
    print(f"\n--- {label} ---")
    if result.get("status") != "ok":
        print(f"  status: {result.get('status')}")
        print(f"  container_id: {result.get('container_id')}")
        print(f"  error_type: {result.get('error_type')}")
        print(f"  error_message: {result.get('error_message')}")
        print(f"  client_observed_seconds: {round(client_observed_seconds, 3)}")
        return
    for key, description in _BENCHMARK_FIELD_ORDER:
        print(f"  {description}: {result.get(key)}")
    print(f"  client_observed_seconds (wall clock, includes network): {round(client_observed_seconds, 3)}")


@app.local_entrypoint()
def benchmark(
    short_audio: str = "audio/test-arabic-short.wav",
    full_audio: str = "audio/test-arabic.wav",
):
    """Test 2 realtime benchmark.

    Runs THREE inference requests through a single `MuseTalkRealtimeWorker`
    instance so Modal reuses the same warm container for all of them
    (no separate `modal run` processes, which could land on different
    containers):

      1. short audio (first request on this container: cold start + first
         inference are both included here since @modal.enter() runs once,
         lazily, before the first method call is dispatched)
      2. short audio again immediately afterwards (expected: SAME warm
         container, no model/avatar reload)
      3. full test-arabic.wav (only run if both short requests succeeded)

    container_id is generated once in @modal.enter() per container
    lifecycle, so comparing it across requests proves whether the same
    container served them.
    """
    print("\nMuseTalk realtime worker benchmark (Test 2)")
    print("=============================================")
    print("Reusing a single worker instance for all requests below, so ")
    print("Modal can route them to the same warm container.\n")

    worker = MuseTalkRealtimeWorker()

    print("=== Request #1: short audio (expected: cold container start) ===")
    t0 = time.time()
    result1 = worker.infer.remote(
        audio_path=short_audio,
        output_name="realtime-short-first.mp4",
    )
    elapsed1 = time.time() - t0
    _print_benchmark_result("Request #1 (short, cold)", result1, elapsed1)
    if result1.get("status") != "ok":
        raise SystemExit("Request #1 (short audio) failed; aborting benchmark. See error above.")

    print("\n=== Request #2: short audio again (expected: SAME warm container) ===")
    t0 = time.time()
    result2 = worker.infer.remote(
        audio_path=short_audio,
        output_name="realtime-short-warm.mp4",
    )
    elapsed2 = time.time() - t0
    _print_benchmark_result("Request #2 (short, warm)", result2, elapsed2)
    if result2.get("status") != "ok":
        raise SystemExit("Request #2 (short audio, warm) failed; aborting benchmark. See error above.")

    same_container = (
        result1.get("container_id") is not None
        and result1.get("container_id") == result2.get("container_id")
    )
    print("\n--- Container identity check ---")
    print(f"  request #1 container_id: {result1.get('container_id')}")
    print(f"  request #2 container_id: {result2.get('container_id')}")
    print(f"  SAME container reused for #1 and #2: {same_container}")

    print("\n=== Request #3: full audio (test-arabic.wav) ===")
    t0 = time.time()
    result3 = worker.infer.remote(
        audio_path=full_audio,
        output_name="realtime-full.mp4",
    )
    elapsed3 = time.time() - t0
    _print_benchmark_result("Request #3 (full audio)", result3, elapsed3)

    print("\n--- Container identity check (full audio request) ---")
    print(f"  request #3 container_id: {result3.get('container_id')}")
    print(
        "  SAME container reused for #1 and #3: "
        f"{result1.get('container_id') == result3.get('container_id')}"
    )

    print("\nBenchmark complete. Output videos written to calm-avatar-assets:/output/")
    for result in (result1, result2, result3):
        if result.get("status") == "ok":
            print(f"  {result['output_path']}")
    print("\nDownload with e.g.:")
    print(
        "  modal volume get --env main calm-avatar-assets "
        'output/realtime-short-first.mp4 .'
    )
    print(
        "  modal volume get --env main calm-avatar-assets "
        'output/realtime-short-warm.mp4 .'
    )
    print(
        "  modal volume get --env main calm-avatar-assets "
        'output/realtime-full.mp4 .'
    )


@app.local_entrypoint()
def benchmark_optimize(
    short_audio: str = "audio/test-arabic-short.wav",
):
    """Optimize-the-warm-path benchmark.

    Reuses a SINGLE `MuseTalkRealtimeWorker` instance for every request
    below, so all of them land on the same warm container (verified via
    container_id), isolating the pipeline/batch-size/fps differences from
    any cold-start effects:

      1. baseline pipeline_mode, 25 fps  ("before": per-frame deepcopy +
         disk PNGs + two ffmpeg passes)
      2. optimized pipeline_mode, 25 fps ("after": no deepcopy, in-RAM
         frames piped directly into a single ffmpeg encode+mux pass)
         -> saved to output/realtime-25fps.mp4
      3. optimized pipeline_mode, 18 fps -> output/realtime-18fps.mp4
      4. optimized pipeline_mode, batch_size in (4, 8, 16), 25 fps, to
         isolate the effect of VAE/UNet batch size alone

    Does not touch the MuseTalk model, the face-parsing/blending algorithm,
    or the cached avatar artifacts -- only the per-request Python/I-O glue
    around UNet+VAE is varied.
    """
    print("\nMuseTalk realtime warm-path optimization benchmark")
    print("=====================================================")
    print("Reusing a single worker instance for all requests below, so ")
    print("Modal can route them to the same warm container.\n")

    worker = MuseTalkRealtimeWorker()

    def run(label, **kwargs):
        t0 = time.time()
        result = worker.infer.remote(audio_path=short_audio, **kwargs)
        elapsed = time.time() - t0
        _print_benchmark_result(label, result, elapsed)
        if result.get("status") != "ok":
            raise SystemExit(f"{label} failed; aborting benchmark. See error above.")
        return result

    print("=== Step 1: BASELINE pipeline, 25 fps (per-frame disk I/O) ===")
    baseline_25 = run(
        "Baseline, 25 fps",
        output_name="realtime-baseline-25fps.mp4",
        pipeline_mode="baseline",
        fps=25,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    print("\n=== Step 2: OPTIMIZED pipeline, 25 fps (in-RAM, piped ffmpeg) ===")
    optimized_25 = run(
        "Optimized, 25 fps",
        output_name="realtime-25fps.mp4",
        pipeline_mode="optimized",
        fps=25,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    print("\n=== Step 3: OPTIMIZED pipeline, 18 fps (in-RAM, piped ffmpeg) ===")
    optimized_18 = run(
        "Optimized, 18 fps",
        output_name="realtime-18fps.mp4",
        pipeline_mode="optimized",
        fps=18,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    # Step 1's baseline run was also this container's very FIRST inference
    # call, which pays a one-time CUDA/cuDNN warmup cost (kernel selection
    # for these exact tensor shapes) on top of the disk-I/O cost being
    # measured -- the same first-call spike seen earlier in
    # audio_feature_extraction_seconds. Re-running baseline now, once the
    # container is fully warm, isolates the disk-I/O cost alone for a fair
    # apples-to-apples comparison against the (already warm) optimized runs.
    print("\n=== Step 3b: BASELINE pipeline again, now fully warm (fair comparison) ===")
    baseline_25_warm = run(
        "Baseline, 25 fps (warm)",
        output_name="realtime-baseline-25fps-warm.mp4",
        pipeline_mode="baseline",
        fps=25,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    print("\n=== Step 4: OPTIMIZED pipeline, batch-size sweep (25 fps) ===")
    batch_results = {}
    for bs in (4, 8, 16):
        batch_results[bs] = run(
            f"Optimized, 25 fps, batch_size={bs}",
            output_name=f"realtime-batchsize-{bs}-tmp.mp4",
            pipeline_mode="optimized",
            fps=25,
            batch_size=bs,
        )

    ids = {
        "baseline_25_cold_first_call": baseline_25.get("container_id"),
        "optimized_25": optimized_25.get("container_id"),
        "optimized_18": optimized_18.get("container_id"),
        "baseline_25_warm": baseline_25_warm.get("container_id"),
        **{f"batch_{bs}": r.get("container_id") for bs, r in batch_results.items()},
    }
    all_same = len(set(ids.values())) == 1
    print("\n--- Container identity check (all steps) ---")
    for label, cid in ids.items():
        print(f"  {label}: {cid}")
    print(f"  SAME container reused for ALL steps: {all_same}")

    print("\n--- Summary: baseline (cold first call) vs optimized (25 fps) ---")
    print(f"{'metric':<32}{'baseline':>14}{'optimized':>14}")
    for key, label in (
        ("audio_feature_extraction_seconds", "audio features (s)"),
        ("unet_inference_seconds", "UNet (s)"),
        ("compositing_seconds", "compositing (s)"),
        ("imwrite_seconds", "imwrite (s)"),
        ("video_encoding_muxing_seconds", "encode/mux (s)"),
        ("vae_decode_compositing_seconds", "vae+composite (s)"),
        ("total_request_seconds", "total request (s)"),
        ("realtime_factor", "realtime factor"),
        ("num_output_frames", "output frames"),
    ):
        print(f"{label:<32}{baseline_25.get(key):>14}{optimized_25.get(key):>14}")

    print("\n--- Summary: baseline (fully warm) vs optimized (25 fps) [fair comparison] ---")
    print(f"{'metric':<32}{'baseline':>14}{'optimized':>14}")
    for key, label in (
        ("audio_feature_extraction_seconds", "audio features (s)"),
        ("unet_inference_seconds", "UNet (s)"),
        ("compositing_seconds", "compositing (s)"),
        ("imwrite_seconds", "imwrite (s)"),
        ("video_encoding_muxing_seconds", "encode/mux (s)"),
        ("vae_decode_compositing_seconds", "vae+composite (s)"),
        ("total_request_seconds", "total request (s)"),
        ("realtime_factor", "realtime factor"),
        ("num_output_frames", "output frames"),
    ):
        print(f"{label:<32}{baseline_25_warm.get(key):>14}{optimized_25.get(key):>14}")

    print("\n--- Summary: optimized 25 fps vs 18 fps ---")
    print(f"{'metric':<32}{'25 fps':>14}{'18 fps':>14}")
    for key, label in (
        ("num_output_frames", "output frames"),
        ("unet_inference_seconds", "UNet (s)"),
        ("vae_decode_seconds", "VAE decode (s)"),
        ("compositing_seconds", "compositing (s)"),
        ("video_encoding_muxing_seconds", "encode/mux (s)"),
        ("total_request_seconds", "total request (s)"),
        ("realtime_factor", "realtime factor"),
        ("effective_generated_fps", "effective gen FPS"),
    ):
        print(f"{label:<32}{optimized_25.get(key):>14}{optimized_18.get(key):>14}")

    print("\n--- Summary: batch size sweep (optimized, 25 fps) ---")
    print(f"{'metric':<32}{'bs=4':>12}{'bs=8':>12}{'bs=16':>12}")
    for key, label in (
        ("unet_inference_seconds", "UNet (s)"),
        ("vae_decode_seconds", "VAE decode (s)"),
        ("total_request_seconds", "total request (s)"),
        ("realtime_factor", "realtime factor"),
    ):
        row = "".join(f"{batch_results[bs].get(key):>12}" for bs in (4, 8, 16))
        print(f"{label:<32}{row}")

    print("\nOutputs written to calm-avatar-assets:/output/:")
    print("  output/realtime-baseline-25fps.mp4")
    print("  output/realtime-25fps.mp4")
    print("  output/realtime-18fps.mp4")
    for bs in (4, 8, 16):
        print(f"  output/realtime-batchsize-{bs}-tmp.mp4")


@app.local_entrypoint()
def benchmark_compositor(
    short_audio: str = "audio/test-arabic-short.wav",
):
    """NumPy/OpenCV compositing benchmark (single optimization stage).

    Reuses a SINGLE `MuseTalkRealtimeWorker` instance for all three
    requests below (same warm container, verified via container_id), same
    2-second audio, same 25 fps, same batch_size, same MuseTalk model/VAE:

      1. CURRENT              -- pipeline_mode="optimized" (PIL-based
         get_image_blending, in-RAM frames, piped ffmpeg, default preset)
         -> output/realtime-25fps-before-compositor.mp4
      2. NUMPY/OPENCV         -- pipeline_mode="numpy" (direct NumPy/OpenCV
         blend using the alpha precomputed once per avatar frame during
         avatar preparation), default ffmpeg preset
         -> output/realtime-25fps-numpy-compositor.mp4
      3. NUMPY/OPENCV + ULTRAFAST -- same as (2), plus encode_preset=
         "ultrafast"
         -> output/realtime-25fps-numpy-ultrafast.mp4

    Resolution and fps are identical across all three; only the compositing
    implementation and (for step 3) the ffmpeg encode preset vary.
    """
    print("\nMuseTalk NumPy/OpenCV compositing benchmark")
    print("==============================================")
    print("Reusing a single worker instance for all requests below, so ")
    print("Modal can route them to the same warm container.\n")

    worker = MuseTalkRealtimeWorker()

    def run(label, **kwargs):
        t0 = time.time()
        result = worker.infer.remote(audio_path=short_audio, fps=25, batch_size=DEFAULT_BATCH_SIZE, **kwargs)
        elapsed = time.time() - t0
        _print_benchmark_result(label, result, elapsed)
        if result.get("status") != "ok":
            print(f"\n*** {label} FAILED -- see error above. Continuing to report remaining steps. ***")
        return result

    print("=== CURRENT: pipeline_mode=optimized (PIL get_image_blending) ===")
    current = run(
        "CURRENT (PIL compositor)",
        output_name="realtime-25fps-before-compositor.mp4",
        pipeline_mode="optimized",
    )

    # This container's very FIRST inference call (above) also pays a
    # one-time CUDA/cuDNN warmup cost (kernel selection for these exact
    # tensor shapes) on top of the compositing cost being measured -- the
    # same first-call spike documented in the prior optimization stage.
    # Re-running CURRENT now, fully warm, isolates the compositing-only
    # cost for a fair total-time/RTF comparison against the (already warm)
    # NUMPY runs. Saved under a throwaway name; the three required
    # deliverable files are unaffected.
    print("\n=== CURRENT again, now fully warm (fair total-time comparison) ===")
    current_warm = run(
        "CURRENT (PIL compositor, warm)",
        output_name="realtime-25fps-before-compositor-warm-tmp.mp4",
        pipeline_mode="optimized",
    )

    print("\n=== NUMPY/OPENCV COMPOSITING: pipeline_mode=numpy ===")
    numpy_result = run(
        "NUMPY/OPENCV compositor",
        output_name="realtime-25fps-numpy-compositor.mp4",
        pipeline_mode="numpy",
    )

    print("\n=== NUMPY/OPENCV + ULTRAFAST ENCODE ===")
    numpy_ultrafast = run(
        "NUMPY/OPENCV + ultrafast",
        output_name="realtime-25fps-numpy-ultrafast.mp4",
        pipeline_mode="numpy",
        encode_preset="ultrafast",
    )

    results = {
        "CURRENT": current_warm,
        "NUMPY/OPENCV": numpy_result,
        "NUMPY+ULTRAFAST": numpy_ultrafast,
    }

    print("\n--- Container identity check (includes cold first CURRENT call) ---")
    ids = {"CURRENT (cold, 1st call)": current.get("container_id"), **{l: r.get("container_id") for l, r in results.items()}}
    for label, cid in ids.items():
        print(f"  {label}: {cid}")
    print(f"  SAME container reused for ALL steps: {len(set(ids.values())) == 1}")

    any_failed = any(r.get("status") != "ok" for r in [current] + list(results.values()))
    if any_failed:
        print("\n--- Errors ---")
        for label, r in {"CURRENT (cold, 1st call)": current, **results}.items():
            if r.get("status") != "ok":
                print(f"  {label}: {r.get('error_type')}: {r.get('error_message')}")
        raise SystemExit("One or more compositor benchmark steps failed; see errors above.")

    print("\n--- Summary: CURRENT (fully warm) vs NUMPY/OPENCV vs NUMPY+ULTRAFAST ---")
    print("(CURRENT here is the fair, fully-warm re-run -- excludes the one-time")
    print(" CUDA/cuDNN warmup paid by this container's very first call.)")
    print(f"{'metric':<28}{'CURRENT':>16}{'NUMPY/OPENCV':>16}{'NUMPY+ULTRA':>16}")
    for key, label in (
        ("unet_inference_seconds", "UNet (s)"),
        ("vae_decode_seconds", "VAE decode (s)"),
        ("compositing_seconds", "compositing (s)"),
        ("video_encoding_muxing_seconds", "encoding (s)"),
        ("total_request_seconds", "total (s)"),
        ("realtime_factor", "RTF"),
        ("num_output_frames", "frame count"),
        ("peak_gpu_memory_mb", "peak GPU mem (MB)"),
    ):
        row = "".join(f"{results[label2].get(key):>16}" for label2 in ("CURRENT", "NUMPY/OPENCV", "NUMPY+ULTRAFAST"))
        print(f"{label:<28}{row}")

    compositing_speedup = current_warm["compositing_seconds"] / numpy_result["compositing_seconds"] \
        if numpy_result["compositing_seconds"] else None
    total_speedup = current_warm["total_request_seconds"] / numpy_ultrafast["total_request_seconds"] \
        if numpy_ultrafast["total_request_seconds"] else None
    print("\n--- Speedups (fair, fully-warm CURRENT vs NUMPY) ---")
    print(f"  compositing: CURRENT -> NUMPY/OPENCV: {round(compositing_speedup, 2) if compositing_speedup else 'n/a'}x")
    print(f"  total request: CURRENT -> NUMPY+ULTRAFAST: {round(total_speedup, 2) if total_speedup else 'n/a'}x")

    print("\nOutputs written to calm-avatar-assets:/output/ (verify visual equivalence manually):")
    print("  output/realtime-25fps-before-compositor.mp4")
    print("  output/realtime-25fps-numpy-compositor.mp4")
    print("  output/realtime-25fps-numpy-ultrafast.mp4")
