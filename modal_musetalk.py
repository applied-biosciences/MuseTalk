import modal

APP_NAME = "calm-musetalk"

app = modal.App(APP_NAME)

models = modal.Volume.from_name(
    "calm-musetalk-models",
    create_if_missing=True,
)

avatars = modal.Volume.from_name(
    "calm-avatar-assets",
    create_if_missing=True,
)


# -------------------------------------------------------------------
# Lightweight CPU API image
# -------------------------------------------------------------------

api_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("fastapi[standard]")
)


# -------------------------------------------------------------------
# MuseTalk GPU image
# -------------------------------------------------------------------

musetalk_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "curl",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
        "libsndfile1",
    )
    .run_commands(
        "git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk",

        "python -m pip install --upgrade pip setuptools wheel",

        "python -m pip install "
        "torch==2.0.1 "
        "torchvision==0.15.2 "
        "torchaudio==2.0.2 "
        "--index-url https://download.pytorch.org/whl/cu118",

        "python -m pip install "
        "-r /opt/MuseTalk/requirements.txt",

        "python -m pip install "
        "--no-cache-dir "
        "-U openmim",

        "mim install mmengine",

        'mim install "mmcv==2.0.1"',

        'mim install "mmdet==3.1.0"',

        # chumpy (a transitive dependency of mmpose) has a legacy setup.py
        # that breaks under pip's default build isolation. Installing it
        # explicitly with --no-build-isolation first works around this.
        "python -m pip install --no-build-isolation chumpy",

        'mim install "mmpose==1.1.0"',
    )
    .env(
        {
            "PYTHONPATH": "/opt/MuseTalk",
            "FFMPEG_PATH": "/usr/bin",
        }
    )
)


# -------------------------------------------------------------------
# Model weights
# -------------------------------------------------------------------

MODELS_DIR = "/models"

# Downloading is pure network/disk work, so it runs on cheap CPU
# containers rather than on the L4 worker.
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.34,<1.0",
        "gdown>=5.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Weights hosted on HuggingFace, laid out exactly as MuseTalk's own
# download_weights.sh produces:
#     (repo_id, filename_in_repo, destination_subdir, required)
#
# `required` is False only for files that inference does not need:
# latentsync_syncnet.pt is used for the sync loss during training.
HF_FILES = [
    ("TMElyralab/MuseTalk", "musetalk/musetalk.json", "", True),
    ("TMElyralab/MuseTalk", "musetalk/pytorch_model.bin", "", True),
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "", True),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "", True),
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae", True),
    (
        "stabilityai/sd-vae-ft-mse",
        "diffusion_pytorch_model.bin",
        "sd-vae",
        True,
    ),
    ("openai/whisper-tiny", "config.json", "whisper", True),
    ("openai/whisper-tiny", "pytorch_model.bin", "whisper", True),
    ("openai/whisper-tiny", "preprocessor_config.json", "whisper", True),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "dwpose", True),
    ("ByteDance/LatentSync", "latentsync_syncnet.pt", "syncnet", False),
]

# Weights fetched over plain HTTP:
#     (url, destination_path_relative_to_MODELS_DIR, required)
URL_FILES = [
    (
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "face-parse-bisent/resnet18-5c106cde.pth",
        True,
    ),
]

# The face parsing checkpoint is only published on Google Drive:
#     (drive_file_id, destination_path_relative_to_MODELS_DIR, required)
GDRIVE_FILES = [
    (
        "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
        "face-parse-bisent/79999_iter.pth",
        True,
    ),
]


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models},
    timeout=5400,
    cpu=4,
)
def download_models(force: bool = False):
    """
    Populate the calm-musetalk-models volume with every weight MuseTalk
    needs (roughly 8 GB in total).

    Safe to re-run: files that already exist are skipped unless
    force=True, so an interrupted run can simply be repeated.

    Returns a plain dict of strings so the result stays picklable for a
    local client that has none of these libraries installed.
    """
    import os
    import shutil
    import traceback
    import urllib.request

    downloaded = []
    skipped = []
    failed = []

    def destination(rel_path):
        return os.path.join(MODELS_DIR, rel_path)

    def already_present(rel_path):
        path = destination(rel_path)
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def record_failure(rel_path, required, exc):
        traceback.print_exc()
        failed.append(
            {
                "path": rel_path,
                "required": required,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    # --- HuggingFace -----------------------------------------------
    from huggingface_hub import hf_hub_download

    for repo_id, filename, subdir, required in HF_FILES:
        rel_path = os.path.join(subdir, filename) if subdir else filename

        if not force and already_present(rel_path):
            skipped.append(rel_path)
            continue

        local_dir = os.path.join(MODELS_DIR, subdir) if subdir else MODELS_DIR

        try:
            os.makedirs(local_dir, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
            downloaded.append(rel_path)
            print(f"downloaded {rel_path} ({repo_id})")
        except Exception as exc:
            record_failure(rel_path, required, exc)

    # --- Direct HTTP -----------------------------------------------
    for url, rel_path, required in URL_FILES:
        if not force and already_present(rel_path):
            skipped.append(rel_path)
            continue

        target = destination(rel_path)
        partial = f"{target}.incomplete"

        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with urllib.request.urlopen(url) as response:
                with open(partial, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            os.replace(partial, target)
            downloaded.append(rel_path)
            print(f"downloaded {rel_path} ({url})")
        except Exception as exc:
            record_failure(rel_path, required, exc)

    # --- Google Drive ----------------------------------------------
    import gdown

    for file_id, rel_path, required in GDRIVE_FILES:
        if not force and already_present(rel_path):
            skipped.append(rel_path)
            continue

        target = destination(rel_path)

        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # gdown returns None instead of raising when Drive refuses
            # to serve the file (quota, changed sharing, HTML warning).
            result = gdown.download(id=file_id, output=target, quiet=False)
            if not result or not already_present(rel_path):
                raise RuntimeError(
                    "gdown could not retrieve the file; download it "
                    "manually and upload it with "
                    f"`modal volume put calm-musetalk-models <file> {rel_path}`"
                )
            downloaded.append(rel_path)
            print(f"downloaded {rel_path} (google drive {file_id})")
        except Exception as exc:
            record_failure(rel_path, required, exc)

    models.commit()

    # --- Verify ----------------------------------------------------
    expected = [
        (
            os.path.join(subdir, filename) if subdir else filename,
            required,
        )
        for _, filename, subdir, required in HF_FILES
    ]
    expected += [(rel_path, required) for _, rel_path, required in URL_FILES]
    expected += [(rel_path, required) for _, rel_path, required in GDRIVE_FILES]

    present = {}
    missing_required = []
    missing_optional = []

    for rel_path, required in expected:
        path = destination(rel_path)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            present[rel_path] = f"{os.path.getsize(path) / 1e6:.1f} MB"
        elif required:
            missing_required.append(rel_path)
        else:
            missing_optional.append(rel_path)

    summary = {
        "status": "complete" if not missing_required else "incomplete",
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "present": present,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }

    print(
        f"{len(present)}/{len(expected)} weight files present "
        f"({len(downloaded)} downloaded, {len(skipped)} already there, "
        f"{len(failed)} failed)"
    )

    return summary


# -------------------------------------------------------------------
# CPU health endpoint
# -------------------------------------------------------------------

@app.function(image=api_image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "message": "MuseTalk service is deployed",
    }


# -------------------------------------------------------------------
# CPU streaming endpoint
# -------------------------------------------------------------------

# Rendered videos are served straight off the assets volume by a small
# CPU container, so a webpage can point a <video> tag at Modal instead
# of downloading the file first.
STREAM_DIR = "/avatars"

# Only files under these subdirectories of the volume are servable, so
# source footage and input audio stay private.
OUTPUT_SUBDIR = "output"
STREAMABLE_SUBDIRS = (OUTPUT_SUBDIR,)

# 1 MiB keeps memory flat while still being large enough that a typical
# player fetches a whole clip in a handful of reads.
STREAM_CHUNK_SIZE = 1024 * 1024

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


@app.function(
    image=api_image,
    # Mounted read-write rather than read-only so that reload() below can
    # refresh the mount; nothing in this function writes to the volume.
    volumes={STREAM_DIR: avatars},
    scaledown_window=300,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def stream():
    """
    Serve rendered videos over HTTP with byte-range support.

    Range support is what makes the files usable from a webpage: without
    it seeking is impossible and Safari refuses to play the video at all.
    """
    import os
    import re

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    web = FastAPI(title=f"{APP_NAME} media")

    # Lets a <video> tag on any origin play these files. Narrow
    # allow_origins to your own domain to stop other sites embedding them.
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Range"],
        expose_headers=[
            "Accept-Ranges",
            "Content-Length",
            "Content-Range",
        ],
    )

    def refresh_volume():
        """
        A container sees the volume as it was when the container started,
        so videos rendered since then are invisible until we reload.
        """
        try:
            avatars.reload()
        except Exception as exc:
            # A stale listing is preferable to failing the request.
            print(f"volume reload failed: {type(exc).__name__}: {exc}")

    def resolve(rel_path: str) -> str:
        """
        Turn a request path into a file on the volume, rejecting anything
        that escapes the whitelisted subdirectories (e.g. "../source/x").
        """
        candidate = os.path.normpath(
            os.path.join(STREAM_DIR, rel_path.lstrip("/"))
        )

        roots = [os.path.join(STREAM_DIR, d) for d in STREAMABLE_SUBDIRS]
        if not any(candidate.startswith(root + os.sep) for root in roots):
            raise HTTPException(status_code=404, detail="not found")

        if not os.path.isfile(candidate):
            raise HTTPException(status_code=404, detail="not found")

        return candidate

    def serve(path: str, request: Request) -> Response:
        size = os.path.getsize(path)
        extension = os.path.splitext(path)[1].lower()

        start = 0
        end = size - 1
        status_code = 200
        headers = {
            "accept-ranges": "bytes",
            "cache-control": "public, max-age=3600",
        }
        media_type = MEDIA_TYPES.get(extension, "application/octet-stream")

        # Only "bytes=<first>-[<last>]" is honoured. Suffix ranges and
        # multi-range requests fall through to a normal 200, which is a
        # legal response and one every player handles.
        range_header = request.headers.get("range", "")
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())

        if match:
            start = int(match.group(1))
            if match.group(2):
                end = min(int(match.group(2)), size - 1)

            if start > end or start >= size:
                return Response(
                    status_code=416,
                    headers={
                        "content-range": f"bytes */{size}",
                        "accept-ranges": "bytes",
                    },
                )

            status_code = 206
            headers["content-range"] = f"bytes {start}-{end}/{size}"

        length = end - start + 1
        headers["content-length"] = str(length)

        # Players probe with HEAD before streaming; answering with the
        # headers alone avoids reading the file for nothing.
        if request.method == "HEAD":
            return Response(
                status_code=status_code,
                headers=headers,
                media_type=media_type,
            )

        def chunks():
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(STREAM_CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            chunks(),
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )

    @web.get("/")
    def index():
        return {
            "app": APP_NAME,
            "videos": "/videos",
            "stream": "/video/{name}",
            "player": "/player/{name}",
        }

    @web.get("/videos")
    def list_videos():
        refresh_volume()

        root = os.path.join(STREAM_DIR, OUTPUT_SUBDIR)
        if not os.path.isdir(root):
            return {"videos": []}

        videos = []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            extension = os.path.splitext(name)[1].lower()

            if os.path.isfile(path) and extension in MEDIA_TYPES:
                videos.append(
                    {
                        "name": name,
                        "size_mb": round(os.path.getsize(path) / 1e6, 2),
                        "url": f"/video/{name}",
                    }
                )

        return {"videos": videos}

    @web.api_route("/video/{name:path}", methods=["GET", "HEAD"])
    def video(name: str, request: Request):
        refresh_volume()
        return serve(resolve(os.path.join(OUTPUT_SUBDIR, name)), request)

    @web.get("/player/{name:path}", response_class=HTMLResponse)
    def player(name: str):
        # Confirms the file exists before handing back a page that would
        # otherwise render an empty player.
        refresh_volume()
        resolve(os.path.join(OUTPUT_SUBDIR, name))

        return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{name}</title>
    <style>
      body {{ margin: 0; background: #111; display: grid;
              place-items: center; min-height: 100vh; }}
      video {{ max-width: 90vw; max-height: 90vh; }}
    </style>
  </head>
  <body>
    <video src="/video/{name}" controls autoplay playsinline></video>
  </body>
</html>
"""

    return web


# -------------------------------------------------------------------
# L4 GPU worker
# -------------------------------------------------------------------
REPO_DIR = "/opt/MuseTalk"
AVATARS_DIR = "/avatars"
OUTPUT_SUBDIR = "output"

# MuseTalk resolves several checkpoint paths relative to the current
# working directory rather than accepting them as arguments:
#
#   musetalk/utils/utils.py            -> models/<vae_type>
#   musetalk/utils/face_parsing        -> ./models/face-parse-bisent/...
#   musetalk/utils/preprocessing.py    -> ./models/dwpose/dw-ll_ucoco_384.pth
#
# Symlinking the mounted volume to <repo>/models and running with the
# repo as cwd therefore satisfies all of them at once.
REPO_MODELS_LINK = f"{REPO_DIR}/models"

# Weights that must exist before inference can start. Checked up front so
# a missing file produces a clear message instead of a failure deep inside
# the upstream script.
REQUIRED_FOR_INFERENCE = [
    "musetalkV15/unet.pth",
    "musetalkV15/musetalk.json",
    "sd-vae/config.json",
    "sd-vae/diffusion_pytorch_model.bin",
    "whisper/config.json",
    "whisper/pytorch_model.bin",
    "whisper/preprocessor_config.json",
    "dwpose/dw-ll_ucoco_384.pth",
    "face-parse-bisent/79999_iter.pth",
    "face-parse-bisent/resnet18-5c106cde.pth",
]


@app.cls(
    image=musetalk_image,
    gpu="L4",
    volumes={
        "/models": models,
        "/avatars": avatars,
    },
    timeout=3600,
    scaledown_window=300,
)
class MuseTalkWorker:

    @modal.enter()
    def start_container(self):
        """
        Runs once whenever Modal starts a new GPU container.

        Verifies CUDA and the MuseTalk repository, points the repository's
        expected models directory at the mounted volume, and warms the
        model weights so they are loaded once per container instead of
        once per inference call (see _get_models/_get_face_parser).
        """
        import os
        import traceback

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.isdir(REPO_DIR):
            raise RuntimeError(
                "MuseTalk repository is missing from the container image"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available inside the GPU container"
            )

        self._link_models()

        # VAE/FaceParsing/DWPose all resolve their weight paths relative
        # to the current working directory (e.g. "./models/..."), so
        # anchor the process here once for the container's lifetime
        # instead of relying on a per-call subprocess's cwd=REPO_DIR.
        os.chdir(REPO_DIR)

        self._model_cache = {}
        self._face_parser_cache = {}

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")

        # Warm the default (and overwhelmingly common) combination now,
        # so even the first inference call on a fresh container skips
        # model loading rather than paying for it on the hot path. This
        # also imports musetalk.utils.preprocessing, whose DWPose and
        # FaceAlignment models load once at module-import time.
        try:
            self._get_models(version="v15", use_float16=True)
            self._get_face_parser(left_cheek_width=90, right_cheek_width=90)
        except Exception:
            # Don't fail container startup if warming fails (e.g. weights
            # not yet downloaded into the volume); inference() will
            # surface a clear error, or load lazily on the first call.
            traceback.print_exc()

    def _link_models(self):
        """
        Make <repo>/models resolve to the mounted models volume.

        Anything already sitting at that path came from the image (the
        upstream repo ships an empty placeholder), never from the volume,
        so replacing it cannot destroy downloaded weights.
        """
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

    def _get_models(self, version: str, use_float16: bool) -> dict:
        """
        Return the VAE/UNet/Whisper bundle for (version, use_float16),
        loading and caching it on first use.

        Loading these from disk to GPU is the dominant cost of a cold
        inference call, so a warm container should never pay it twice
        for the same combination.
        """
        key = (version, use_float16)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached

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

        for label, path in (
            ("unet weights", unet_model_path),
            ("unet config", unet_config),
        ):
            if not os.path.isfile(path):
                raise RuntimeError(f"{version} {label} missing: {path}")

        device = self.device
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

        bundle = {
            "vae": vae,
            "unet": unet,
            "pe": pe,
            "whisper": whisper,
            "audio_processor": audio_processor,
            "weight_dtype": weight_dtype,
        }
        self._model_cache[key] = bundle
        print(f"loaded models for version={version} use_float16={use_float16}")
        return bundle

    def _get_face_parser(self, left_cheek_width: int, right_cheek_width: int):
        """
        Return the FaceParsing (BiSeNet) instance for these cheek widths,
        loading and caching it on first use.
        """
        key = (left_cheek_width, right_cheek_width)
        cached = self._face_parser_cache.get(key)
        if cached is not None:
            return cached

        from musetalk.utils.face_parsing import FaceParsing

        face_parser = FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        )
        self._face_parser_cache[key] = face_parser
        print(
            "loaded face parser for "
            f"left_cheek_width={left_cheek_width} "
            f"right_cheek_width={right_cheek_width}"
        )
        return face_parser

    def _run_inference(
        self,
        *,
        source_video: str,
        source_audio: str,
        output_vid_name: str,
        result_dir: str,
        version: str,
        batch_size: int,
        fps: int,
        extra_margin: int,
        parsing_mode: str,
        bbox_shift: int,
        models: dict,
        face_parser,
    ) -> str:
        """
        Run MuseTalk generation in-process using already-loaded models.

        Mirrors the per-task body of upstream scripts/inference.py::main,
        but takes plain arguments and reuses cached model objects instead
        of reloading them from disk on every call. scripts/inference.py
        itself is left untouched and keeps working standalone.

        Returns the absolute path to the produced video.
        """
        import copy
        import glob
        import os
        import shutil
        import subprocess
        import time

        import cv2
        import numpy as np
        import torch

        from musetalk.utils.blending import get_image
        from musetalk.utils.preprocessing import (
            coord_placeholder,
            get_landmark_and_bbox,
        )
        from musetalk.utils.utils import datagen, get_file_type, get_video_fps

        device = self.device
        vae = models["vae"]
        unet = models["unet"]
        pe = models["pe"]
        audio_processor = models["audio_processor"]
        whisper = models["whisper"]
        weight_dtype = models["weight_dtype"]

        timesteps = torch.tensor([0], device=device)

        input_basename = os.path.splitext(os.path.basename(source_video))[0]
        audio_basename = os.path.splitext(os.path.basename(source_audio))[0]

        temp_dir = os.path.join(result_dir, version)
        os.makedirs(temp_dir, exist_ok=True)

        result_img_save_path = os.path.join(
            temp_dir, f"{input_basename}_{audio_basename}"
        )
        crop_coord_save_path = os.path.join(
            result_dir, "..", input_basename + ".pkl"
        )
        os.makedirs(result_img_save_path, exist_ok=True)

        output_path = os.path.join(temp_dir, output_vid_name)
        save_dir_full = None

        with torch.no_grad():
            # Extract frames from the avatar video.
            file_type = get_file_type(source_video)
            if file_type == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = (
                    f"ffmpeg -v fatal -i {source_video} "
                    f"-start_number 0 {save_dir_full}/%08d.png"
                )
                os.system(cmd)
                input_img_list = sorted(
                    glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]"))
                )
                video_fps = get_video_fps(source_video)
            elif file_type == "image":
                input_img_list = [source_video]
                video_fps = fps
            elif os.path.isdir(source_video):
                input_img_list = glob.glob(
                    os.path.join(source_video, "*.[jpJP][pnPN]*[gG]")
                )
                input_img_list = sorted(
                    input_img_list,
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
                )
                video_fps = fps
            else:
                raise ValueError(
                    f"{source_video} should be a video file, an image "
                    "file or a directory of images"
                )

            # Extract audio features.
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(
                source_audio
            )
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

            # Extract landmarks/bboxes and encode avatar frames to latents.
            print("Extracting landmarks...")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(
                    crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                )
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)

            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Batch inference.
            musetalk_started = time.time()
            print(
                f"[diag] MuseTalk inference starting: "
                f"input_basename={input_basename} audio_basename={audio_basename} "
                f"num_frames={len(frame_list_cycle)} batch_size={batch_size} "
                f"at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print("Starting inference")
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )

            res_frame_list = []
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                pred_latents = unet.model(
                    latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
                ).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)

            print(
                f"[diag] MuseTalk inference finished: "
                f"generated {len(res_frame_list)} frames in "
                f"{time.time() - musetalk_started:.2f}s at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Blend generated faces back into the original frames.
            print("Padding generated images to original video size")
            for i, res_frame in enumerate(res_frame_list):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2 = y2 + extra_margin
                    y2 = min(y2, ori_frame.shape[0])
                try:
                    res_frame = cv2.resize(
                        res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                    )
                except Exception:
                    continue

                if version == "v15":
                    combine_frame = get_image(
                        ori_frame,
                        res_frame,
                        [x1, y1, x2, y2],
                        mode=parsing_mode,
                        fp=face_parser,
                    )
                else:
                    combine_frame = get_image(
                        ori_frame, res_frame, [x1, y1, x2, y2], fp=face_parser
                    )
                cv2.imwrite(
                    f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame
                )

        # Mux frames + audio into the final video.
        temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
        cmd_img2video = (
            f"ffmpeg -y -v warning -r {video_fps} -f image2 "
            f"-i {result_img_save_path}/%08d.png "
            f"-vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
        )
        os.system(cmd_img2video)

        silent_video_exists = os.path.isfile(temp_vid_path)
        silent_video_size = (
            os.path.getsize(temp_vid_path) if silent_video_exists else 0
        )
        print(
            f"[diag] silent video path={temp_vid_path} "
            f"exists={silent_video_exists} size_bytes={silent_video_size}"
        )

        audio_exists = os.path.isfile(source_audio)
        audio_size = os.path.getsize(source_audio) if audio_exists else 0
        print(
            f"[diag] input audio path={source_audio} "
            f"exists={audio_exists} size_bytes={audio_size}"
        )

        cmd_combine_audio = (
            f"ffmpeg -y -v warning -i {source_audio} "
            f"-i {temp_vid_path} {output_path}"
        )
        print(f"[diag] FFmpeg command: {cmd_combine_audio}")
        ffmpeg_started = time.time()
        print(f"[diag] FFmpeg starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        ffmpeg_result = subprocess.run(
            cmd_combine_audio,
            shell=True,
            capture_output=True,
            text=True,
        )
        print(
            f"[diag] FFmpeg finished: returncode={ffmpeg_result.returncode} "
            f"in {time.time() - ffmpeg_started:.2f}s"
        )
        if ffmpeg_result.returncode != 0:
            print(f"[diag] FFmpeg stderr:\n{ffmpeg_result.stderr}")

        final_mp4_exists = os.path.isfile(output_path)
        final_mp4_size = os.path.getsize(output_path) if final_mp4_exists else 0
        print(
            f"[diag] final mp4 path={output_path} "
            f"exists={final_mp4_exists} size_bytes={final_mp4_size}"
        )

        shutil.rmtree(result_img_save_path, ignore_errors=True)
        if os.path.isfile(temp_vid_path):
            os.remove(temp_vid_path)
        if save_dir_full and os.path.isdir(save_dir_full):
            shutil.rmtree(save_dir_full, ignore_errors=True)
        if os.path.isfile(crop_coord_save_path):
            os.remove(crop_coord_save_path)

        return output_path

    @modal.method()
    def inference(
        self,
        video_path: str = "source/saudi-female.mp4",
        audio_path: str = "audio/sakinah-greeting.wav",
        output_name: str = None,
        version: str = "v15",
        batch_size: int = 8,
        fps: int = 25,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        bbox_shift: int = 0,
        use_float16: bool = True,
    ):
        """
        Generate a lip-synced video from an avatar video and an audio clip.

        video_path and audio_path are relative to the calm-avatar-assets
        volume. The result is written back to that volume under output/
        and the returned path is relative to it as well.

        Returns a plain dict of primitives so the result stays picklable
        for a local client without torch installed.
        """
        import os
        import shutil
        import time
        import traceback
        import uuid

        started = time.time()

        try:
            source_video = os.path.join(AVATARS_DIR, video_path)
            source_audio = os.path.join(AVATARS_DIR, audio_path)

            # Pick up files uploaded since this container started.
            avatars.reload()
            models.reload()

            for label, path in (
                ("video", source_video),
                ("audio", source_audio),
            ):
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"{label} not found in the calm-avatar-assets "
                        f"volume: {path}"
                    )

            missing = [
                rel_path
                for rel_path in REQUIRED_FOR_INFERENCE
                if not os.path.isfile(os.path.join(MODELS_DIR, rel_path))
            ]
            if missing:
                raise RuntimeError(
                    "Missing weights in the calm-musetalk-models volume: "
                    f"{missing}. Run `modal run modal_musetalk.py::download`."
                )

            if version not in ("v15", "v1"):
                raise ValueError(
                    f"version must be 'v15' or 'v1', got {version!r}"
                )

            self._link_models()

            # Scratch space. Landmark caching writes a pickle to
            # result_dir/../, so give it a private parent directory.
            run_id = uuid.uuid4().hex[:8]
            work_dir = f"/tmp/musetalk/{run_id}"
            result_dir = os.path.join(work_dir, "results")
            os.makedirs(result_dir, exist_ok=True)

            stem = output_name or (
                f"{os.path.splitext(os.path.basename(video_path))[0]}"
                f"_{os.path.splitext(os.path.basename(audio_path))[0]}"
            )
            if stem.endswith(".mp4"):
                stem = stem[: -len(".mp4")]
            output_vid_name = f"{stem}.mp4"

            # Reuses cached models loaded once per container (see
            # start_container/_get_models/_get_face_parser) instead of
            # spawning a fresh `python -m scripts.inference` subprocess
            # that would reload every weight from disk on every call.
            model_bundle = self._get_models(
                version=version, use_float16=use_float16
            )
            face_parser = self._get_face_parser(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width,
            )

            produced = self._run_inference(
                source_video=source_video,
                source_audio=source_audio,
                output_vid_name=output_vid_name,
                result_dir=result_dir,
                version=version,
                batch_size=batch_size,
                fps=fps,
                extra_margin=extra_margin,
                parsing_mode=parsing_mode,
                bbox_shift=bbox_shift,
                models=model_bundle,
                face_parser=face_parser,
            )

            if not os.path.isfile(produced):
                raise RuntimeError("Inference produced no output video")

            output_rel_path = os.path.join(OUTPUT_SUBDIR, output_vid_name)
            destination = os.path.join(AVATARS_DIR, output_rel_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(produced, destination)
            avatars.commit()

            size_bytes = os.path.getsize(destination)
            print(
                f"[diag] final mp4 copied to volume: path={destination} "
                f"exists={os.path.isfile(destination)} size_bytes={size_bytes}"
            )
            shutil.rmtree(work_dir, ignore_errors=True)

            result = {
                "status": "ok",
                "app": APP_NAME,
                "version": version,
                "video_path": video_path,
                "audio_path": audio_path,
                "output_path": output_rel_path,
                "output_volume": "calm-avatar-assets",
                "output_size_mb": round(size_bytes / 1e6, 2),
                "duration_seconds": round(time.time() - started, 1),
            }
            print(f"[diag] returning result to calm-musetalk-bridge: {result}")
            return result
        except Exception as exc:
            # Return a plain-string error rather than letting an exception
            # that may reference torch objects escape to a local client
            # that cannot deserialize them.
            traceback.print_exc()
            return {
                "status": "error",
                "app": APP_NAME,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "duration_seconds": round(time.time() - started, 1),
            }

    @modal.method()
    def status(self):
        import traceback

        try:
            import torch

            return {
                "status": "ready",
                "app": APP_NAME,
                "device": str(self.device),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "torch_version": str(torch.__version__),
                "models_mounted": True,
                "avatars_mounted": True,
            }
        except Exception as exc:
            # Print the full traceback server-side and return a plain-string
            # error instead of letting an unpicklable exception (e.g. one
            # referencing the torch module in its frame locals) escape,
            # which the local client cannot deserialize without torch installed.
            traceback.print_exc()
            return {
                "status": "error",
                "app": APP_NAME,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }


# -------------------------------------------------------------------
# Laptop-side test command
# -------------------------------------------------------------------

@app.local_entrypoint()
def download(force: bool = False):
    print("Downloading MuseTalk weights into the models volume...")

    summary = download_models.remote(force=force)

    for rel_path, size in sorted(summary["present"].items()):
        print(f"  {rel_path:<48} {size}")

    for failure in summary["failed"]:
        marker = "required" if failure["required"] else "optional"
        print(f"  FAILED ({marker}) {failure['path']}: {failure['error']}")

    if summary["missing_optional"]:
        print(f"Missing optional: {summary['missing_optional']}")

    if summary["status"] != "complete":
        raise SystemExit(
            f"Missing required weights: {summary['missing_required']}"
        )

    print("All required weights are in place.")


@app.local_entrypoint()
def generate(
    video_path: str = "source/saudi-female.mp4",
    audio_path: str = "audio/sakinah-greeting.wav",
    output_name: str = None,
    version: str = "v15",
):
    print(f"Generating lip-synced video from {video_path} + {audio_path}...")

    result = MuseTalkWorker().inference.remote(
        video_path=video_path,
        audio_path=audio_path,
        output_name=output_name,
        version=version,
    )

    for key, value in result.items():
        print(f"  {key}: {value}")

    if result["status"] != "ok":
        raise SystemExit("Inference failed")

    print(
        "Download it with: modal volume get calm-avatar-assets "
        f"{result['output_path']} ."
    )


@app.local_entrypoint()
def test_gpu():
    print("Starting MuseTalk L4 worker...")

    result = MuseTalkWorker().status.remote()

    print("GPU worker result:")
    print(result)



@app.function(
    image=api_image,
    volumes={"/avatars": avatars},
)
@modal.fastapi_endpoint(method="GET")
def test_saudi_female():
    from pathlib import Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    video = Path("/avatars/source/saudi-female.mp4")

    if not video.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video does not exist: {video}",
        )

    return FileResponse(
        path=str(video),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
        },
    )
