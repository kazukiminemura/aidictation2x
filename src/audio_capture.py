import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1


class AudioRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._is_recording = False
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def start(self) -> None:
        if self.is_recording:
            return

        def callback(indata, frames, time, status):  # noqa: ANN001, ARG001
            if status:
                # Callback内ではUIへ例外を投げず継続する。
                return
            self._audio_queue.put(indata.copy())

        with self._lock:
            self._audio_queue = queue.Queue()
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate_hz,
                channels=self.config.channels,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
            self._is_recording = True

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            return np.array([], dtype=np.float32)

        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._is_recording = False

        chunks = []
        while not self._audio_queue.empty():
            chunks.append(self._audio_queue.get())

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks, axis=0).reshape(-1)
