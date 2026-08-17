"""Video-to-screen progressive delivery validation (engineering test only).

This is a NEW, additive Modal endpoint. It does not modify:
  - modal_musetalk.py / MuseTalkWorker (batch, validated)
  - modal_musetalk_realtime.py / MuseTalkRealtimeWorker.infer (validated
    warm realtime pipeline + NumPy compositor)

It reuses the exact same warm-loading helpers (_load_model_bundle,
_load_face_parser, _prepare_avatar) and the exact same NumPy/OpenCV
blending math validated in MuseTalkRealtimeWorker's "numpy" pipeline_mode,
but instead of writing a complete .mp4 to the volume and returning once
finished, it streams fragmented-MP4 (fMP4) bytes to the HTTP client AS
EACH BATCH of frames is generated, so the browser can start receiving
(and, if its player supports progressive fMP4 playback, start playing)
video before generation is 100% complete.

Run:
    modal deploy --env main modal_musetalk_stream.py
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

# This file imports the sibling `modal_musetalk_realtime` module (for its
# warm-loading helpers) at top level, so -- same reasoning as
# modal_musetalk_realtime.py bundling modal_musetalk.py -- it must also be
# bundled into the image explicitly for remote container reconstruction.
# realtime_image already bundles modal_musetalk.py; this only adds
# modal_musetalk_realtime.py on top, leaving both untouched.
stream_image = realtime_image.add_local_python_source("modal_musetalk_realtime")

DEFAULT_STREAM_FPS = 25  # keep 25 FPS, per instructions
GOP_SIZE = DEFAULT_BATCH_SIZE  # one keyframe per compositing batch -> one fMP4 fragment per batch


@app.cls(
    image=stream_image,
    gpu="L4",
    volumes={MODELS_DIR: models_volume, AVATARS_DIR: avatars},
    timeout=3600,
    scaledown_window=300,
)
class MuseTalkStreamTestWorker:
    """Same warm-container pattern as MuseTalkRealtimeWorker, plus a
    progressive-streaming ASGI endpoint for this validation test only."""

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
            print(f"stream worker: avatar ready (source={source})")
        except Exception as exc:
            traceback.print_exc()
            self._startup_error = f"{type(exc).__name__}: {exc}"

        print(f"stream worker container_id={self.container_id} ready in {_time.time() - enter_started:.2f}s")

    def _generate_fmp4(self, audio_path: str, t1: float):
        """Sync generator: composites frames batch-by-batch (NumPy compositor,
        identical math to MuseTalkRealtimeWorker's "numpy" pipeline_mode) and
        pipes them into a persistent ffmpeg process emitting fragmented MP4,
        yielding output bytes as soon as ffmpeg produces them -- not waiting
        for the whole clip to finish encoding.

        FastAPI/Starlette runs sync generators passed to StreamingResponse in
        a worker thread automatically, so this function is allowed to block
        (GPU calls, subprocess I/O) without stalling the event loop.
        """
        import os
        import subprocess

        import cv2
        import numpy as np
        import torch

        from musetalk.utils.utils import datagen

        t2 = time.time()
        print(f"T2 (inference begins)={t2}")

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

        # Fragmented MP4: -g/-keyint_min force a keyframe at the start of
        # every compositing batch, and frag_keyframe tells ffmpeg to close
        # an output fragment at each keyframe -- so a new, independently
        # playable fragment is flushed roughly every `batch_size` frames
        # instead of only once at the very end. Baseline profile keeps the
        # MSE codec string simple/universal for the browser side of this
        # test. This only affects encoder/container settings for THIS test
        # endpoint; it does not change MuseTalk, the compositor, or pixels.
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{frame_w}x{frame_h}", "-r", str(DEFAULT_STREAM_FPS),
            "-i", "pipe:0",
            "-i", source_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-profile:v", "baseline", "-level", "3.0",
            "-g", str(GOP_SIZE), "-keyint_min", str(GOP_SIZE), "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-ar", "44100",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # A dedicated reader thread drains ffmpeg's stdout continuously.
        # Without this, writing many frames to stdin while nobody reads
        # stdout can deadlock once the OS pipe buffer fills (classic
        # subprocess two-pipe deadlock).
        chunk_queue: "queue.Queue" = queue.Queue()

        def _drain_stdout():
            try:
                while True:
                    data = proc.stdout.read(65536)
                    if not data:
                        break
                    chunk_queue.put(data)
            finally:
                chunk_queue.put(None)  # sentinel: stdout closed (ffmpeg done)

        reader_thread = threading.Thread(target=_drain_stdout, daemon=True)
        reader_thread.start()

        first_frame_logged = False
        first_chunk_yielded = False
        num_output_frames = 0

        def _drain_ready_chunks(block=False):
            nonlocal first_chunk_yielded
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
                if not first_chunk_yielded:
                    first_chunk_yielded = True
                    t4 = time.time()
                    print(f"T4 (first video data leaves Modal)={t4}")
            return out_chunks

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

                        if not first_frame_logged:
                            first_frame_logged = True
                            t3 = time.time()
                            print(f"T3 (first generated frame available)={t3}")

                        proc.stdin.write(out.tobytes())
                        num_output_frames += 1

                    # Give ffmpeg a chance to flush this batch's fragment and
                    # forward anything it has already produced, so the HTTP
                    # response keeps making progress while later batches are
                    # still being generated (rather than only yielding at the
                    # very end).
                    for chunk in _drain_ready_chunks():
                        if chunk is None:
                            break
                        yield chunk

            proc.stdin.close()

            # Drain everything else ffmpeg still has to flush (trailer,
            # final fragment) after we've finished feeding it frames.
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
                print(f"ffmpeg stderr tail: {stderr_tail}")
            t8 = time.time()
            print(
                f"T8 (generation+encoding finished server-side)={t8} "
                f"frames={num_output_frames} audio_duration_s={audio_duration_seconds:.3f} "
                f"t1_to_t8_s={t8 - t1:.3f}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse

        web_app = FastAPI(title="MuseTalk progressive stream test")
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
                "app": "musetalk-stream-test",
                "container_id": getattr(self, "container_id", None),
                "stream": "/stream?audio_path=audio/test-arabic-short.wav",
            }

        @web_app.get("/stream")
        def stream(audio_path: str = "audio/test-arabic-short.wav"):
            # T1: this Modal endpoint has received the request. Known
            # synchronously before any generation work starts, so it can be
            # sent immediately as a response header (unlike T2-T4, which
            # only become known after streaming has already begun and are
            # therefore reported via container logs instead).
            t1 = time.time()
            print(f"T1 (Modal endpoint received request)={t1} audio_path={audio_path}")

            return StreamingResponse(
                self._generate_fmp4(audio_path, t1),
                media_type="video/mp4",
                headers={
                    "X-T1-Epoch-Seconds": repr(t1),
                    "X-Container-Id": str(getattr(self, "container_id", "")),
                    "Cache-Control": "no-store",
                },
            )

        return web_app
