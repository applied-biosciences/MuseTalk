"""Latency investigation for time-to-first-visible-frame (T6 - T0).

This is a NEW, additive, SEPARATE endpoint from the already-accepted
modal_musetalk_stream.py baseline, which is left completely untouched and
remains deployed/reachable throughout this experiment (per instructions:
"keep the existing working implementation available as the baseline and
make experimental changes separately").

It does not change the MuseTalk model, VAE, avatar, resolution, lip-sync
algorithm, the NumPy compositor, or 25 FPS -- only two isolated,
independently-toggleable things about the video encoding/streaming glue:

  1. `eager_drain`: whether the ffmpeg-stdout-to-HTTP-response draining
     loop checks for ready bytes after every composited frame (True) or
     only after every full compositing batch (False, matching the
     baseline's current behavior).
  2. `first_fragment_frames`: how many frames the FIRST fMP4 fragment
     contains, via an explicit `-force_key_frames` expression forcing an
     early keyframe at that frame index. All fragments AFTER the first
     keep the same GOP=DEFAULT_BATCH_SIZE cadence as the baseline
     (`first_fragment_frames=DEFAULT_BATCH_SIZE` exactly reproduces the
     baseline's fragmentation structure, so it doubles as a same-code-path
     "baseline" comparison row for TEST A/B).

Run:
    modal deploy --env main modal_musetalk_stream_experiment.py
"""

import queue
import threading
import time

import modal

from modal_musetalk import AVATARS_DIR, MODELS_DIR, REPO_DIR, app, avatars
from modal_musetalk_realtime import (
    DEFAULT_AVATAR_ID,
    DEFAULT_AVATAR_VIDEO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BBOX_SHIFT,
    DEFAULT_EXTRA_MARGIN,
    DEFAULT_LEFT_CHEEK_WIDTH,
    DEFAULT_PARSING_MODE,
    DEFAULT_RIGHT_CHEEK_WIDTH,
    DEFAULT_VERSION,
    _ensure_models_symlink,
    _load_face_parser,
    _load_model_bundle,
    _prepare_avatar,
    realtime_image,
)
from modal_musetalk_realtime import models_volume

# Modal deploys snapshot the ENTIRE app's function set from whatever the
# deployed entrypoint file imports at module load time. Importing (but not
# modifying) modal_musetalk_stream here -- purely for this side effect --
# keeps the already-accepted baseline endpoint (MuseTalkStreamTestWorker)
# registered and reachable alongside this experiment on every subsequent
# `modal deploy` of either file, satisfying "keep the existing working
# implementation available as the baseline." The baseline file itself is
# not changed in any way.
import modal_musetalk_stream  # noqa: F401

stream_image = realtime_image.add_local_python_source(
    "modal_musetalk_realtime", "modal_musetalk_stream"
)

DEFAULT_STREAM_FPS = 25  # unchanged
GOP_SIZE = DEFAULT_BATCH_SIZE  # unchanged: GOP cadence after the first fragment


@app.cls(
    image=stream_image,
    gpu="L4",
    volumes={MODELS_DIR: models_volume, AVATARS_DIR: avatars},
    timeout=3600,
    scaledown_window=300,
)
class MuseTalkStreamExperimentWorker:
    """Same warm-container pattern as MuseTalkStreamTestWorker, plus fine-
    grained A1-A6 instrumentation and two independently-toggleable
    experimental knobs (see module docstring)."""

    @modal.enter()
    def start_container(self):
        import os
        import time as _time
        import traceback
        import uuid

        import torch

        enter_started = _time.time()
        self.container_id = uuid.uuid4().hex[:12]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = None
        self.face_parser = None
        self.avatar_cache = {}
        self._startup_error = None

        if not os.path.isdir(REPO_DIR):
            raise RuntimeError("MuseTalk repository is missing from the container image")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the GPU container")

        _ensure_models_symlink()
        os.chdir(REPO_DIR)

        try:
            self.models = _load_model_bundle(device=self.device, version=DEFAULT_VERSION, use_float16=True)
            self.face_parser = _load_face_parser(DEFAULT_LEFT_CHEEK_WIDTH, DEFAULT_RIGHT_CHEEK_WIDTH)

            avatar_data, _, source = _prepare_avatar(
                avatar_id=DEFAULT_AVATAR_ID,
                video_path_abs=os.path.join(AVATARS_DIR, DEFAULT_AVATAR_VIDEO),
                video_path_rel=DEFAULT_AVATAR_VIDEO,
                version=DEFAULT_VERSION,
                bbox_shift=DEFAULT_BBOX_SHIFT,
                extra_margin=DEFAULT_EXTRA_MARGIN,
                parsing_mode=DEFAULT_PARSING_MODE,
                fps=DEFAULT_STREAM_FPS,
                vae=self.models["vae"],
                face_parser=self.face_parser,
            )
            self.avatar_cache[DEFAULT_AVATAR_ID] = avatar_data
            print(f"experiment worker: avatar ready (source={source})")
        except Exception as exc:
            traceback.print_exc()
            self._startup_error = f"{type(exc).__name__}: {exc}"

        print(f"experiment worker container_id={self.container_id} ready in {_time.time() - enter_started:.2f}s")

    def _generate_fmp4(self, audio_path: str, t1: float, first_fragment_frames: int, eager_drain: bool):
        import os
        import subprocess

        import cv2
        import numpy as np
        import torch

        from musetalk.utils.utils import datagen

        t2 = time.time()
        run_tag = f"[ff={first_fragment_frames} eager={eager_drain}]"
        print(f"{run_tag} A0/T2 (inference begins)={t2}")

        if self._startup_error:
            raise RuntimeError(f"container warm-up failed: {self._startup_error}")

        source_audio = os.path.join(AVATARS_DIR, audio_path)
        avatars.reload()
        if not os.path.isfile(source_audio):
            raise FileNotFoundError(f"audio not found in calm-avatar-assets volume: {source_audio}")

        avatar_data = self.avatar_cache[DEFAULT_AVATAR_ID]
        frame_list_cycle = avatar_data["frame_list_cycle"]
        coord_list_cycle = avatar_data["coord_list_cycle"]
        input_latent_list_cycle = avatar_data["input_latent_list_cycle"]
        alpha_list_cycle = avatar_data["alpha_list_cycle"]

        device = self.device
        vae = self.models["vae"]
        unet = self.models["unet"]
        pe = self.models["pe"]
        whisper = self.models["whisper"]
        audio_processor = self.models["audio_processor"]
        weight_dtype = self.models["weight_dtype"]
        timesteps = torch.tensor([0], device=device)

        whisper_input_features, librosa_length = audio_processor.get_audio_feature(source_audio)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=DEFAULT_STREAM_FPS,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        audio_duration_seconds = librosa_length / 16000.0

        frame_h, frame_w = frame_list_cycle[0].shape[:2]

        # force_key_frames places an extra keyframe at `first_fragment_frames`
        # (in addition to the mandatory keyframe at frame 0), so fragment 0
        # contains exactly `first_fragment_frames` frames. -g/-keyint_min
        # resume the normal GOP_SIZE cadence from that point onward (an
        # automatic keyframe every GOP_SIZE frames after the last one),
        # exactly like the baseline for every fragment after the first.
        # When first_fragment_frames == GOP_SIZE this is a no-op relative to
        # the baseline's plain "-g 8 -keyint_min 8" (the forced keyframe
        # lands exactly where an automatic one already would).
        force_kf_expr = f"expr:eq(n,0)+eq(n,{first_fragment_frames})"
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{frame_w}x{frame_h}", "-r", str(DEFAULT_STREAM_FPS),
            "-i", "pipe:0",
            "-i", source_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-profile:v", "baseline", "-level", "3.0",
            "-force_key_frames", force_kf_expr,
            "-g", str(GOP_SIZE), "-keyint_min", str(GOP_SIZE), "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-ar", "44100",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]
        print(f"{run_tag} ffmpeg_cmd={' '.join(ffmpeg_cmd)}")

        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        chunk_queue: "queue.Queue" = queue.Queue()
        reader_state = {"chunks_seen": 0, "a4": None, "a5": None}

        def _drain_stdout():
            try:
                while True:
                    data = proc.stdout.read(65536)
                    if not data:
                        break
                    now = time.time()
                    reader_state["chunks_seen"] += 1
                    if reader_state["a4"] is None:
                        reader_state["a4"] = now
                        print(f"{run_tag} A4 (first bytes on ffmpeg stdout, size={len(data)})={now}")
                    elif reader_state["a5"] is None:
                        reader_state["a5"] = now
                        print(f"{run_tag} A5 (second stdout read -> first real media fragment, size={len(data)})={now}")
                    chunk_queue.put(data)
            finally:
                chunk_queue.put(None)

        reader_thread = threading.Thread(target=_drain_stdout, daemon=True)
        reader_thread.start()

        state = {"a1": None, "a2": None, "a3": None, "a6": None}

        def _drain_ready_chunks(block=False):
            out_chunks = []
            while True:
                try:
                    item = chunk_queue.get(timeout=0.2 if block and not out_chunks else 0)
                except queue.Empty:
                    break
                if item is None:
                    out_chunks.append(None)
                    break
                out_chunks.append(item)
                if state["a6"] is None:
                    state["a6"] = time.time()
                    print(f"{run_tag} A6 (first bytes yielded by StreamingResponse)={state['a6']}")
            return out_chunks

        num_output_frames = 0
        try:
            with torch.no_grad():
                for whisper_batch, latent_batch in datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list_cycle,
                    batch_size=DEFAULT_BATCH_SIZE,
                    delay_frame=0,
                    device=device,
                ):
                    audio_feature_batch = pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=unet.model.dtype)
                    pred_latents = unet.model(
                        latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
                    ).sample
                    recon = vae.decode_latents(pred_latents)

                    for res_frame in recon:
                        idx = num_output_frames % len(coord_list_cycle)
                        bbox = coord_list_cycle[idx]
                        x1, y1, x2, y2 = bbox
                        try:
                            res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                        except Exception:
                            num_output_frames += 1
                            continue

                        alpha = alpha_list_cycle[idx % len(alpha_list_cycle)]
                        background = frame_list_cycle[idx % len(frame_list_cycle)]
                        out = background.copy()
                        roi = out[y1:y2, x1:x2].astype(np.float32)
                        face_f = res_frame_resized.astype(np.float32)
                        blended = face_f * alpha + roi * (1.0 - alpha)
                        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

                        if state["a1"] is None:
                            state["a1"] = time.time()
                            print(f"{run_tag} A1 (first composited frame available)={state['a1']}")

                        proc.stdin.write(out.tobytes())
                        now = time.time()
                        if state["a2"] is None:
                            state["a2"] = now
                            print(f"{run_tag} A2 (first frame written to ffmpeg stdin)={now}")
                        if num_output_frames == first_fragment_frames and state["a3"] is None:
                            state["a3"] = now
                            print(
                                f"{run_tag} A3 (wrote boundary frame idx={num_output_frames} "
                                f"-> ffmpeg has enough frames for fragment 0)={now}"
                            )

                        num_output_frames += 1

                        if eager_drain:
                            for chunk in _drain_ready_chunks():
                                if chunk is None:
                                    break
                                yield chunk

                    if not eager_drain:
                        for chunk in _drain_ready_chunks():
                            if chunk is None:
                                break
                            yield chunk

            proc.stdin.close()

            while True:
                chunks = _drain_ready_chunks(block=True)
                done = False
                for chunk in chunks:
                    if chunk is None:
                        done = True
                        break
                    yield chunk
                if done:
                    break

            proc.wait(timeout=30)
            reader_thread.join(timeout=5)
            stderr_tail = proc.stderr.read().decode(errors="ignore")[-2000:]
            if proc.returncode != 0:
                print(f"{run_tag} ffmpeg stderr tail: {stderr_tail}")
            t8 = time.time()
            print(
                f"{run_tag} T8 (generation+encoding finished server-side)={t8} "
                f"frames={num_output_frames} audio_duration_s={audio_duration_seconds:.3f} "
                f"t1_to_t8_s={t8 - t1:.3f} reader_chunks_seen={reader_state['chunks_seen']}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse

        web_app = FastAPI(title="MuseTalk progressive stream latency experiment")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "HEAD", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-T1-Epoch-Seconds", "X-Container-Id"],
        )

        @web_app.get("/")
        def index():
            return {
                "app": "musetalk-stream-experiment",
                "container_id": getattr(self, "container_id", None),
                "stream": "/stream?audio_path=audio/test-arabic-short.wav&first_fragment_frames=8&eager_drain=true",
            }

        @web_app.get("/stream")
        def stream(
            audio_path: str = "audio/test-arabic-short.wav",
            first_fragment_frames: int = 8,
            eager_drain: bool = True,
        ):
            t1 = time.time()
            print(
                f"[ff={first_fragment_frames} eager={eager_drain}] T1 (Modal endpoint received request)={t1} "
                f"audio_path={audio_path}"
            )

            return StreamingResponse(
                self._generate_fmp4(audio_path, t1, first_fragment_frames, eager_drain),
                media_type="video/mp4",
                headers={
                    "X-T1-Epoch-Seconds": repr(t1),
                    "X-Container-Id": str(getattr(self, "container_id", "")),
                    "X-First-Fragment-Frames": str(first_fragment_frames),
                    "X-Eager-Drain": str(eager_drain),
                    "Cache-Control": "no-store",
                },
            )

        return web_app
