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

        Verifies CUDA and the MuseTalk repository, then points the
        repository's expected models directory at the mounted volume.
        """
        import os
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

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")

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
        import subprocess
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

            if version == "v15":
                unet_model_path = f"{MODELS_DIR}/musetalkV15/unet.pth"
                unet_config = f"{MODELS_DIR}/musetalkV15/musetalk.json"
            elif version == "v1":
                unet_model_path = f"{MODELS_DIR}/musetalk/pytorch_model.bin"
                unet_config = f"{MODELS_DIR}/musetalk/musetalk.json"
            else:
                raise ValueError(
                    f"version must be 'v15' or 'v1', got {version!r}"
                )

            for label, path in (
                ("unet weights", unet_model_path),
                ("unet config", unet_config),
            ):
                if not os.path.isfile(path):
                    raise RuntimeError(f"{version} {label} missing: {path}")

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

            # Upstream only accepts tasks through a YAML config file.
            config_path = os.path.join(work_dir, "task.yaml")
            with open(config_path, "w") as handle:
                handle.write(
                    "task_0:\n"
                    f'  video_path: "{source_video}"\n'
                    f'  audio_path: "{source_audio}"\n'
                    f"  bbox_shift: {bbox_shift}\n"
                )

            command = [
                "python",
                "-m",
                "scripts.inference",
                "--inference_config", config_path,
                "--result_dir", result_dir,
                "--unet_model_path", unet_model_path,
                "--unet_config", unet_config,
                "--whisper_dir", f"{MODELS_DIR}/whisper",
                "--version", version,
                "--output_vid_name", output_vid_name,
                "--batch_size", str(batch_size),
                "--fps", str(fps),
                "--extra_margin", str(extra_margin),
                "--parsing_mode", parsing_mode,
                "--left_cheek_width", str(left_cheek_width),
                "--right_cheek_width", str(right_cheek_width),
                "--bbox_shift", str(bbox_shift),
                "--ffmpeg_path", "/usr/bin",
            ]
            if use_float16:
                command.append("--use_float16")

            print("running:", " ".join(command))

            completed = subprocess.run(
                command,
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
            )

            # Stream the child output into this container's logs so the
            # Modal dashboard shows progress and tracebacks.
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)

            # scripts.inference catches per-task exceptions, prints them and
            # still exits 0, so a zero return code does not mean success.
            # The produced file is the only trustworthy signal.
            produced = os.path.join(result_dir, version, output_vid_name)

            if not os.path.isfile(produced):
                tail = "\n".join(
                    (completed.stdout or "").splitlines()[-15:]
                    + (completed.stderr or "").splitlines()[-15:]
                )
                raise RuntimeError(
                    "Inference produced no output video "
                    f"(exit code {completed.returncode}). Last log lines:\n"
                    f"{tail}"
                )

            output_rel_path = os.path.join(OUTPUT_SUBDIR, output_vid_name)
            destination = os.path.join(AVATARS_DIR, output_rel_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(produced, destination)
            avatars.commit()

            size_bytes = os.path.getsize(destination)
            shutil.rmtree(work_dir, ignore_errors=True)

            return {
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
