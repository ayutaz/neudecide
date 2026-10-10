"""Web UI for trying NeuDecide live with a microphone.

    uv run --group webui examples/webui_example.py    # then open http://127.0.0.1:8000

The browser captures the microphone (or plays a WAV file) and streams raw samples over a
WebSocket. This server splits them into utterances with the same loudness gate as
realtime_example.py and sends back the tool calls for each one, along with the audio
levels the page draws.
"""

import asyncio
import base64
import io
import json
import time
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from neudecide import NeuDecide
from neudecide.audio import resample
from realtime_example import FRAME_MS, MIN_THRESHOLD_DB, Segmenter, rms_db, threshold_from

EXAMPLES = Path(__file__).parent
STATIC = EXAMPLES / "webui"


def wav_bytes(x, sample_rate):
    """A float waveform as a 16-bit PCM WAV file's contents."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def check_tools(tools):
    if not isinstance(tools, list) or not all(isinstance(t, dict) and "name" in t for t in tools):
        raise ValueError('tools must be a JSON list of objects, each with a "name"')
    return tools


class Session:
    """One stream from the browser: samples in, events out.

    The first `calibrate` seconds are taken as background noise and set the speech
    threshold (unless one is given); after that, frames go through a Segmenter."""

    def __init__(self, sample_rate, tools, margin=12.0, silence_ms=500, threshold=None):
        self.sample_rate = int(sample_rate)
        self.frame = round(self.sample_rate * FRAME_MS / 1000)
        self.pending = np.zeros(0, dtype=np.float32)
        self.tools = check_tools(tools)
        self.margin, self.silence_ms, self.manual_threshold = margin, silence_ms, threshold
        self.calibrate()

    def calibrate(self, seconds=1.0):
        self.background = []
        self.noise = None
        self.background_frames = max(1, round(seconds * 1000 / FRAME_MS))
        self.segmenter = None
        return {"type": "calibrating", "seconds": seconds}

    def configure(self, settings):
        """Applies changed settings; an utterance in progress is dropped."""
        if "tools" in settings:
            self.tools = check_tools(settings["tools"])
        self.margin = float(settings.get("margin", self.margin))
        self.silence_ms = int(settings.get("silenceMs", self.silence_ms))
        if "threshold" in settings:
            t = settings["threshold"]
            self.manual_threshold = None if t is None else float(t)
        if self.manual_threshold is None and self.noise is None:
            if self.segmenter is not None:  # was on a manual threshold: measure the room now
                return self.calibrate()
            return None  # still calibrating; the new margin applies when that's done
        return self.listen()

    @property
    def threshold(self):
        if self.manual_threshold is not None:
            return self.manual_threshold
        return max(self.noise + self.margin, MIN_THRESHOLD_DB)

    def listen(self):
        self.segmenter = Segmenter(self.threshold, silence_ms=self.silence_ms)
        return {"type": "listening", "threshold": self.threshold, "noise": self.noise}

    def feed(self, samples):
        """Yields events for the page. Those of type "utterance" carry the audio in
        "audio" and still need the model."""
        self.pending = np.concatenate([self.pending, samples.astype(np.float32, copy=False)])
        n = len(self.pending) // self.frame
        frames = self.pending[: n * self.frame].reshape(n, self.frame)
        self.pending = self.pending[n * self.frame :]
        for frame in frames:
            db = max(float(rms_db(frame)), -120.0)
            if self.segmenter is None:
                yield {"type": "level", "db": db, "speech": False}
                self.background.append(frame)
                if len(self.background) >= self.background_frames:
                    _, self.noise = threshold_from(self.background, self.margin)
                    yield self.listen()
                continue
            was_active = self.segmenter.active
            utterance = self.segmenter.push(frame)
            yield {"type": "level", "db": db, "speech": self.segmenter.active}
            if self.segmenter.active and not was_active:
                yield {"type": "speech"}
            if utterance is not None:
                yield {"type": "utterance", "audio": utterance}
            elif was_active and not self.segmenter.active:
                yield {"type": "ignored"}  # too little speech: a click or a bump


def create_app(model, default_tools, save_dir=None):
    """save_dir: if set, each utterance is kept there as NNNN.wav (as the model heard
    it), with its result appended to results.jsonl."""
    app = FastAPI()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/capture-worklet.js")
    def worklet():
        return FileResponse(STATIC / "capture-worklet.js", media_type="text/javascript")

    @app.get("/api/tools")
    def tools():
        return JSONResponse(default_tools)

    async def recognise(ws, session, audio):
        await ws.send_json({"type": "thinking"})
        start = time.perf_counter()
        calls = await asyncio.to_thread(
            model.generate, audio, session.tools, sample_rate=session.sample_rate
        )
        result = {
            "calls": calls,
            "seconds": len(audio) / session.sample_rate,
            "ms": (time.perf_counter() - start) * 1000,
        }
        # what the model heard: resampled to its rate, which also keeps this small
        wav = wav_bytes(resample(audio, session.sample_rate, model.sample_rate), model.sample_rate)
        if save_dir is not None:
            name = f"{len(list(save_dir.glob('*.wav'))):04d}.wav"
            (save_dir / name).write_bytes(wav)
            record = {
                "file": name,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                **result,
                "capture_rate": session.sample_rate,
                "threshold": session.threshold,
                "noise": session.noise,
                "tools": session.tools,
            }
            with open(save_dir / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        await ws.send_json(
            {"type": "result", **result, "audio": base64.b64encode(wav).decode("ascii")}
        )

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        """Text messages are JSON: {"type": "start", "sampleRate", "tools", "margin",
        "silenceMs", "threshold"} first, then "settings" (any of the same fields) or
        "calibrate". Binary messages are little-endian float32 mono samples."""
        await ws.accept()
        session = None
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if session is None:
                        continue
                    for event in list(session.feed(np.frombuffer(message["bytes"], "<f4"))):
                        if event["type"] == "utterance":
                            await recognise(ws, session, event["audio"])
                        else:
                            await ws.send_json(event)
                    continue
                data = json.loads(message["text"])
                try:
                    if data["type"] == "start":
                        session = Session(
                            data["sampleRate"],
                            data.get("tools", default_tools),
                            margin=data.get("margin", 12.0),
                            silence_ms=data.get("silenceMs", 500),
                            threshold=data.get("threshold"),
                        )
                        if session.manual_threshold is not None:
                            event = session.listen()
                        else:
                            event = session.calibrate()
                    elif session is None:
                        continue
                    elif data["type"] == "settings":
                        event = session.configure(data)
                    elif data["type"] == "calibrate":
                        event = session.calibrate()
                    else:
                        continue
                except (ValueError, KeyError, TypeError) as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
                if event:
                    await ws.send_json(event)
        except WebSocketDisconnect:
            pass

    return app


def main(args):
    with open(args.tools, "r") as f:
        tools = json.load(f)
    model = NeuDecide.from_pretrained(args.repo)
    # the first call sets up constrained decoding; do it now rather than on the first command
    model.generate(np.zeros(model.sample_rate, dtype=np.float32), tools)
    print(f"Open http://{args.host}:{args.port} in a browser.")
    if args.save_dir:
        print(f"Saving each utterance and its result to {args.save_dir}/")
    app = create_app(model, tools, save_dir=args.save_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuDecide web UI")
    parser.add_argument(
        "--tools",
        type=str,
        default=str(EXAMPLES / "tools.json"),
        help="JSON file holding the default list of tool definitions",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Address to serve on")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--save-dir", type=str, help="Keep each utterance (WAV) and its result here, to inspect"
    )
    parser.add_argument(
        "--repo", type=str, default="neuphonic/neudecide", help="Huggingface repo with the model"
    )
    main(parser.parse_args())
