# Telephony Voice Integration

This document summarizes how to feed TTS audio from a telephony/voice platform into the `calm-musetalk` MuseTalk pipeline to generate lip-synced avatar videos. It covers two platforms: **Vapi** (closed-source SaaS) and **Dograh** (open-source, self-hostable).

MuseTalk is a lip-sync/dubbing model, not a text-to-speech engine. It always requires a pre-existing audio clip as input — it cannot synthesize speech or clone a voice from text. Both integrations below exist to get that audio clip into the `calm-avatar-assets` volume so `MuseTalkWorker.inference` can render it.

## Shared pipeline steps

Once you have a WAV file (16kHz mono recommended), both integrations converge on the same two steps:

1. **Upload the audio to the volume:**
   ```python
   import modal
   avatars = modal.Volume.from_name("calm-avatar-assets")
   with avatars.batch_upload() as batch:
       batch.put_file(local_wav_path, f"audio/{call_id}-{turn_index}.wav")
   ```
2. **Trigger inference asynchronously (never block on it):**
   ```python
   from modal_musetalk import MuseTalkWorker
   call_handle = MuseTalkWorker().inference.spawn(
       video_path="source/sakinah-saudi-female-idle.mp4",
       audio_path=f"audio/{call_id}-{turn_index}.wav",
       output_name=f"{call_id}-{turn_index}.mp4",
   )
   # Persist call_handle.object_id to poll/await later.
   ```
   Once `status: ok`, the result is at `output/<name>.mp4` and immediately playable at
   `https://applied-biosciences--calm-musetalk-stream.modal.run/video/<name>.mp4` — no redeploy needed.

Per-call latency is on the order of tens of seconds even after the warm-container optimization (see `modal_musetalk.py`'s `MuseTalkWorker`), so video delivery must always be decoupled from the live call — deliver it afterward via chat/SMS/dashboard rather than gating the conversation on it.

## Vapi integration

Vapi's live call audio isn't otherwise interceptable, so this requires configuring a **custom TTS provider** webhook on the assistant:

1. Set the assistant's `voice` provider to `custom-voice`, pointing at a webhook you host.
2. In that webhook, call your real TTS engine, capture a copy of the raw audio bytes before returning them to Vapi, keyed by call ID and turn index.
3. Normalize the captured audio to 16kHz mono WAV.
4. Run the shared pipeline steps above.
5. Poll the stored `object_id` from a background worker and deliver the resulting video URL once ready.

Fallback (post-call only, not real-time): use the `end-of-call-report` webhook's `call.recordingUrl`, then re-synthesize the assistant's utterances offline with the same TTS voice before running the shared pipeline steps.

## Dograh integration

Dograh is self-hosted and open source, so there are two options:

**Option A (preferred) — hook into Dograh's TTS synthesis step directly**, since you control the backend source:
1. Locate the Voice Synthesizer (TTS) stage in your self-hosted deployment.
2. Add a capture hook right after Dograh calls the configured TTS provider for a turn, writing the synthesized audio to disk keyed by run ID and turn index.
3. Run the shared pipeline steps above.
4. Write the resulting video URL back into the run's `gathered_context`, or fire a workflow webhook, once generation completes.

**Option B (fallback, no source changes)** — use the post-call run recording via the SDK:
```python
from dograh_sdk import DograhClient
client = DograhClient(api_key="YOUR_API_KEY")
run = client.get_run(workflow_id=WORKFLOW_ID, run_id=RUN_ID)
recording_url = run.recording_url
```
The recording contains both parties' audio. If dual-channel, extract the agent's channel with `ffmpeg -map`; if mixed-mono, align `transcript_url` timestamps to isolate the agent's segments, or re-synthesize those segments offline with the same TTS provider/voice. Then run the shared pipeline steps.

Use Option A whenever you can modify the self-hosted deployment; fall back to Option B only if integrating purely through the public SDK/API.
