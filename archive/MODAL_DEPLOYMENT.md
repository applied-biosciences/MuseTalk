# Deploying MuseTalk on Modal

This document covers `modal_musetalk.py`, which runs MuseTalk 1.5 lip-sync
inference on [Modal](https://modal.com) as an app named `calm-musetalk`.

Given an avatar video and an audio clip, it produces a video of the avatar
speaking that audio.

Nothing here modifies MuseTalk itself. The Modal image clones the upstream
repository at build time and mounts the weights from a volume, so this is
purely additive deployment tooling.

## Architecture

The app has three components, deliberately split so cheap work never occupies
a GPU:

| Component | Hardware | Purpose |
| --- | --- | --- |
| `health` | CPU | Public `GET` endpoint reporting that the app is deployed |
| `download_models` | CPU (4 cores) | Populates the model volume with ~8.5 GB of weights |
| `MuseTalkWorker` | L40S preferred, L4 fallback | Runs MuseTalk inference; scales to zero after 5 minutes idle |

Two Modal volumes hold all persistent state, so container images stay lean and
weights survive redeploys:

- **`calm-musetalk-models`** — mounted at `/models`, holds the MuseTalk weights
- **`calm-avatar-assets`** — mounted at `/avatars`, holds inputs and output

```
calm-avatar-assets/
├── source/                     input avatar videos and images
├── audio/                      driving audio clips
└── output/                     generated videos
```

## Prerequisites

- Python 3.9+ locally (the GPU image builds its own Python 3.10)
- A Modal account with a configured token

The local environment only needs the Modal client. Torch, ffmpeg and the mmlab
stack are installed inside the remote images, never locally, so
`requirements-modal.txt` is deliberately separate from MuseTalk's own
`requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-modal.txt
modal token new        # only if `modal config show` has no token_id
```

## Deployment

### 1. Download the model weights (one time, ~8.5 GB)

The volume must be populated before inference will run. This is the long pole,
so start it first:

```bash
modal run modal_musetalk.py::download
```

The function is idempotent — existing files are skipped, so an interrupted run
can simply be repeated. Use `--force` to re-download everything.

It fetches the same file set as `download_weights.sh`, into the same layout:

| Destination | Source |
| --- | --- |
| `musetalk/`, `musetalkV15/` | `TMElyralab/MuseTalk` |
| `sd-vae/` | `stabilityai/sd-vae-ft-mse` |
| `whisper/` | `openai/whisper-tiny` |
| `dwpose/` | `yzd-v/DWPose` |
| `syncnet/` | `ByteDance/LatentSync` (optional, training only) |
| `face-parse-bisent/79999_iter.pth` | Google Drive |
| `face-parse-bisent/resnet18-5c106cde.pth` | `download.pytorch.org` |

Unlike `download_weights.sh`, it uses the real HuggingFace endpoint rather than
`hf-mirror.com`, and enables `hf_transfer` for faster downloads.

Confirm the result:

```bash
modal volume ls calm-musetalk-models
```

**If the Google Drive file fails** (quota limits are the usual cause), the run
reports it explicitly. Download `79999_iter.pth` by hand and upload it:

```bash
modal volume put calm-musetalk-models 79999_iter.pth face-parse-bisent/79999_iter.pth
```

### 2. Upload avatar assets

Paths are relative to the volume root. Quote paths containing spaces — do not
mix quotes with backslash escapes, or the backslashes are treated as literal
characters.

```bash
modal volume put calm-avatar-assets /path/to/avatar.mp4 source/saudi-female.mp4
modal volume put calm-avatar-assets /path/to/clip.wav   audio/greeting.wav
```

Audio should be **16 kHz mono WAV**, which is what the `whisper-tiny` audio
encoder expects. A quick test clip can be generated on macOS with:

```bash
say -v Samantha -o clip.wav --data-format=LEI16@16000 "hello, how can i help you today"
```

### 3. Deploy the app

```bash
modal deploy modal_musetalk.py
```

This publishes the persistent `health` endpoint and registers the worker. The
first deploy builds the CUDA 11.8 + torch 2.0.1 + mmlab image, which takes a
while; later deploys reuse the cached layers.

Use `modal serve modal_musetalk.py` for a hot-reloading development session.

### 4. Generate a video

```bash
modal run modal_musetalk.py::generate \
  --video-path source/saudi-female.mp4 \
  --audio-path audio/greeting.wav \
  --output-name my-result
```

Retrieve the output:

```bash
modal volume get calm-avatar-assets output/my-result.mp4 .
```

## Verifying a deployment

```bash
modal run modal_musetalk.py::test_gpu          # CUDA / GPU / torch versions
curl https://<workspace>--calm-musetalk-health.modal.run
```

`test_gpu` is the cheapest way to confirm the GPU image is healthy without
paying for a full inference run.

## Reference

| Command | Effect |
| --- | --- |
| `modal run modal_musetalk.py::download` | Populate the model volume |
| `modal run modal_musetalk.py::download --force` | Re-download all weights |
| `modal run modal_musetalk.py::generate` | Run inference |
| `modal run modal_musetalk.py::test_gpu` | Report GPU and torch status |
| `modal deploy modal_musetalk.py` | Deploy the app |
| `modal volume ls calm-avatar-assets output` | List generated videos |
| `modal app logs calm-musetalk` | Tail logs |

### Inference parameters

`MuseTalkWorker.inference` accepts these, defaulting to the v1.5
recommendations:

| Parameter | Default | Notes |
| --- | --- | --- |
| `version` | `v15` | `v15` or `v1` |
| `batch_size` | `8` | Raise to trade memory for throughput |
| `extra_margin` | `10` | Extra chin margin, v1.5 only |
| `parsing_mode` | `jaw` | Face blending mode |
| `left/right_cheek_width` | `90` | Cheek blending region |
| `use_float16` | `True` | fp16 inference, well suited to both the L40S and the L4 |
| `bbox_shift` | `0` | **Ignored by v1.5**; affects v1 only |

## Implementation notes

Things that are non-obvious and easy to break:

**Weight paths are resolved relative to the working directory.** Three places
hardcode `./models/...` rather than accepting arguments — the VAE
(`musetalk/utils/utils.py`), the face parser (`musetalk/utils/face_parsing`)
and dwpose (`musetalk/utils/preprocessing.py`). The worker therefore symlinks
`<repo>/models` to the mounted `/models` volume and runs with the repo as its
working directory, which satisfies all of them at once. Only image content is
ever replaced by that symlink, never volume data.

**Inference runs as a subprocess.** `musetalk/utils/preprocessing.py` loads the
dwpose model at *import* time, so importing MuseTalk in-process would trigger
model loading before paths are ready and leave global state in the container.
Invoking `python -m scripts.inference` avoids this.

**A zero exit code does not mean success.** `scripts/inference.py` wraps each
task in `try/except`, prints the error and still exits 0. The worker therefore
treats the existence of the output file as the only trustworthy success signal,
and surfaces the last log lines when it is missing.

**Output length follows the audio, not the video.** MuseTalk cycles the source
frames forward and backward to cover the audio, so a 2.8 s clip against a 10 s
avatar yields a 2.8 s result. Frames are composited back into the original
resolution; only the 256x256 mouth region is regenerated.

**Remote functions return plain dicts.** Errors are caught and returned as
strings rather than raised, because an exception referencing a torch object
cannot be deserialized by a local client that has no torch installed.

### Harmless warnings

```
An error occurred while trying to fetch models/sd-vae: Error no file named
diffusion_pytorch_model.safetensors found in directory models/sd-vae.
```

`diffusers` prefers `.safetensors`, does not find it, and falls back to the
`.bin` file that `download_weights.sh` installs. The VAE loads correctly.

`TF-TRT could not find TensorRT` is likewise irrelevant — TensorRT is not used.

## Performance

The worker requests an L40S first and falls back to an L4 when Modal cannot
allocate one; `test_gpu` reports which GPU was actually assigned. A cold run
on an L4 takes roughly 2.5 minutes end to end for a ~3 second clip; the L40S
is faster.
Landmark extraction dominates; it can be cached to a pickle via
`--use_saved_coord` / `--saved_coord`, which is worth persisting into the
volume if the same avatar is reused repeatedly. Containers stay warm for 5
minutes (`scaledown_window`), so consecutive runs skip startup entirely.
