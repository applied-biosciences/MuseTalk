"""MuseTalk test runner driven by command-line arguments.

Run from the calm-musetalk directory with:

    modal run --env main test_musetalk_interactive.py \\
        --video saudi-female.mp4 \\
        --audio test-arabic.wav \\
        --output saudi-female-arabic-test.mp4

Filenames must already exist in the calm-avatar-assets Modal Volume. You
may pass a bare filename or its Volume-relative path. The job is
submitted immediately by default; pass --no-yes to only print the
resolved job details without running inference.
"""

from pathlib import PurePosixPath

from modal_musetalk import MuseTalkWorker, app


def _volume_path(value: str, default_folder: str) -> str:
    """Accept either a bare filename or a Volume-relative path."""
    normalized = value.lstrip("/")
    path = PurePosixPath(normalized)

    if len(path.parts) == 1:
        path = PurePosixPath(default_folder) / path

    if ".." in path.parts:
        raise ValueError("Paths cannot contain '..'.")

    return str(path)


def _output_name(value: str) -> str:
    """Return a safe MP4 filename rather than a directory path."""
    name = PurePosixPath(value).name

    if not name.lower().endswith(".mp4"):
        name += ".mp4"

    return name


@app.local_entrypoint()
def main(
    video: str,
    audio: str,
    output: str,
    yes: bool = True,
):
    """Submit a MuseTalk inference job.

    Args:
        video: Source video filename or Volume-relative path
            (e.g. "saudi-female.mp4").
        audio: Speech filename or Volume-relative path
            (e.g. "test-arabic.wav").
        output: Output MP4 filename
            (e.g. "saudi-female-arabic-test.mp4").
        yes: Submit the job immediately. Pass --no-yes to only print the
            resolved job details without running inference.
    """
    print("\nGeneric MuseTalk avatar test")
    print("----------------------------")
    print("Files must already exist in the calm-avatar-assets Volume.\n")

    video_path = _volume_path(video, "source")
    audio_path = _volume_path(audio, "audio")
    output_name = _output_name(output)

    print("Submitting MuseTalk job:")
    print(f"  video:  {video_path}")
    print(f"  audio:  {audio_path}")
    print(f"  output: output/{output_name}")

    if not yes:
        raise SystemExit(
            "\nDry run only (--no-yes). Omit that flag to submit the job."
        )

    result = MuseTalkWorker().inference.remote(
        video_path=video_path,
        audio_path=audio_path,
        output_name=output_name,
        version="v15",
    )

    print("\nMuseTalk result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if result.get("status") != "ok":
        raise SystemExit("Inference failed. Check the error above and Modal logs.")

    output_path = result["output_path"]
    print("\nTest completed successfully.")
    print("Download the generated MP4 with:")
    print(
        "  modal volume get --env main calm-avatar-assets "
        f'"{output_path}" "{output_name}"'
    )


