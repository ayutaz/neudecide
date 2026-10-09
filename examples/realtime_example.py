"""Live microphone demo: speak a command, get its tool calls when you stop talking.

Needs sounddevice (`pip install sounddevice`). The microphone stream is split into
utterances by loudness: one starts when the input gets louder than a threshold set from
the room's background level, and ends after a short pause. Each utterance goes to the
model as soon as it ends. This gate matters: the model produces a call even for noise.

    python examples/realtime_example.py                      # microphone
    python examples/realtime_example.py --wav command.wav    # a WAV file in place of the mic
"""

import json
import queue
import time
from collections import deque
from pathlib import Path

import numpy as np

from neudecide import NeuDecide
from neudecide.audio import load_audio

EXAMPLES = Path(__file__).parent
FRAME_MS = 30
MIN_THRESHOLD_DB = -50.0  # never gate below this, even in a near-silent room


def rms_db(frame):
    """Loudness of a frame in dBFS."""
    return 20 * np.log10(np.sqrt(np.mean(np.square(frame))) + 1e-10)


def threshold_from(frames, margin_db):
    """A speech threshold `margin_db` above the median level of background frames."""
    noise = float(np.median([rms_db(f) for f in frames]))
    return max(noise + margin_db, MIN_THRESHOLD_DB), noise


class Segmenter:
    """Splits a stream of FRAME_MS frames into utterances by loudness.

    An utterance starts at the first frame louder than `threshold_db`, with `preroll_ms`
    of the audio before it so the onset isn't clipped, and ends after `silence_ms` of
    quieter frames or at `max_seconds`. Utterances with under `min_speech_ms` of loud
    frames are dropped as clicks and bumps."""

    def __init__(
        self, threshold_db, silence_ms=500, min_speech_ms=200, preroll_ms=300, max_seconds=30
    ):
        def frames(ms):
            return max(1, round(ms / FRAME_MS))

        self.threshold_db = threshold_db
        self.silence_frames = frames(silence_ms)
        self.min_speech_frames = frames(min_speech_ms)
        self.max_frames = frames(max_seconds * 1000)
        self.preroll = deque(maxlen=frames(preroll_ms))
        self.frames = None  # the utterance in progress, if any
        self.speech = self.quiet = 0

    @property
    def active(self):
        return self.frames is not None

    def push(self, frame):
        """Feeds one frame. Returns the utterance it completes, if any, else None."""
        loud = rms_db(frame) > self.threshold_db
        if not self.active:
            if not loud:
                self.preroll.append(frame)
                return None
            self.frames = list(self.preroll)
            self.preroll.clear()
            self.speech = self.quiet = 0
        self.frames.append(frame)
        if loud:
            self.speech += 1
            self.quiet = 0
        else:
            self.quiet += 1
        if self.quiet >= self.silence_frames or len(self.frames) >= self.max_frames:
            return self.flush()
        return None

    def flush(self):
        """Ends the utterance in progress. Returns it, or None if there was none or it
        was too short."""
        frames, speech = self.frames, self.speech
        self.frames = None
        if not frames or speech < self.min_speech_frames:
            return None
        return np.concatenate(frames)


def run(model, tools, utterance, sample_rate):
    start = time.perf_counter()
    calls = model.generate(utterance, tools, sample_rate=sample_rate)
    ms = (time.perf_counter() - start) * 1000
    print(f"  {len(utterance) / sample_rate:.1f} s of speech -> {ms:.0f} ms")
    print(json.dumps(calls, indent=2) if calls else "  (no tool call)")


def listen_wav(model, tools, path, args):
    """Feeds a WAV file through the same segmentation as the microphone, with a second of
    silence after it, as fast as it can."""
    audio, sr = load_audio(path)
    frame = sr * FRAME_MS // 1000
    audio = np.concatenate([audio, np.zeros(sr, dtype=np.float32)])
    frames = [audio[i : i + frame] for i in range(0, len(audio) - frame + 1, frame)]
    if args.threshold is None:
        # no quiet lead-in to calibrate on: take the file's quietest tenth as background
        quiet = sorted(frames, key=rms_db)[: max(1, len(frames) // 10)]
        threshold, noise = threshold_from(quiet, args.margin)
    else:
        threshold, noise = args.threshold, None
    report_threshold(threshold, noise)
    segmenter = Segmenter(threshold, silence_ms=args.silence_ms)
    for f in frames:
        utterance = segmenter.push(f)
        if utterance is not None:
            run(model, tools, utterance, sr)
    utterance = segmenter.flush()
    if utterance is not None:
        run(model, tools, utterance, sr)


def listen_mic(model, tools, args):
    import sounddevice as sd

    sr = int(args.sample_rate or sd.query_devices(args.device, "input")["default_samplerate"])
    frame = sr * FRAME_MS // 1000
    frames = queue.Queue()

    def callback(indata, n, t, status):
        if status:
            print(f"  [audio: {status}]")
        frames.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sr,
        blocksize=frame,
        channels=1,
        dtype="float32",
        device=args.device,
        callback=callback,
    ):
        if args.threshold is None:
            print(f"Measuring background noise for {args.calibrate:g} s, stay quiet...")
            n = max(1, round(args.calibrate * 1000 / FRAME_MS))
            threshold, noise = threshold_from([frames.get() for _ in range(n)], args.margin)
        else:
            threshold, noise = args.threshold, None
        report_threshold(threshold, noise)
        print(f"Listening at {sr} Hz. Speak a command; Ctrl+C to stop.")
        segmenter = Segmenter(threshold, silence_ms=args.silence_ms)
        while True:
            was_active = segmenter.active
            utterance = segmenter.push(frames.get())
            if segmenter.active and not was_active:
                print("* hearing speech...")
            if utterance is not None:
                run(model, tools, utterance, sr)


def report_threshold(threshold, noise):
    if noise is None:
        print(f"Speech threshold: {threshold:.0f} dBFS")
    else:
        print(f"Background {noise:.0f} dBFS -> speech threshold {threshold:.0f} dBFS")


def main(args):
    with open(args.tools, "r") as f:
        tools = json.load(f)
    model = NeuDecide.from_pretrained(args.repo)
    try:
        if args.wav:
            listen_wav(model, tools, args.wav, args)
        else:
            listen_mic(model, tools, args)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuDecide live microphone example")
    parser.add_argument(
        "--tools",
        type=str,
        default=str(EXAMPLES / "tools.json"),
        help="JSON file holding a list of tool definitions",
    )
    parser.add_argument(
        "--wav", type=str, help="Read this WAV file in place of the microphone (for testing)"
    )
    parser.add_argument(
        "--device",
        type=lambda s: int(s) if s.isdigit() else s,
        help="Input device name or index (see `python -m sounddevice`)",
    )
    parser.add_argument(
        "--sample-rate", type=int, help="Capture rate (default: the device's own rate)"
    )
    parser.add_argument(
        "--threshold", type=float, help="Speech threshold in dBFS (default: measured at start)"
    )
    parser.add_argument(
        "--margin", type=float, default=12.0, help="dB above the background level for speech"
    )
    parser.add_argument(
        "--calibrate", type=float, default=1.0, help="Seconds of background noise to measure"
    )
    parser.add_argument(
        "--silence-ms", type=int, default=500, help="Pause that ends an utterance, in ms"
    )
    parser.add_argument(
        "--repo", type=str, default="neuphonic/neudecide", help="Huggingface repo with the model"
    )
    main(parser.parse_args())
