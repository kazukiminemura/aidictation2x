"""Always-on VAD-based audio listener for real-time voice command detection."""
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .audio_capture import AudioConfig


class ContinuousListener:
    """
    Monitors the microphone continuously and fires `on_utterance(audio)` each time
    a complete utterance is detected (speech followed by silence).

    The callback is invoked in a background daemon thread.
    Optional `on_voice_start()` fires when speech is first detected (UI feedback).
    """

    _CHUNK_FRAMES = 480  # 30 ms at 16 kHz

    def __init__(
        self,
        config: AudioConfig,
        on_utterance: Callable[[np.ndarray], None],
        on_voice_start: Optional[Callable[[], None]] = None,
        voice_threshold: float = 0.003,
        silence_duration_s: float = 0.8,
        min_speech_s: float = 0.2,
    ):
        self.config = config
        self.on_utterance = on_utterance
        self.on_voice_start = on_voice_start
        self.voice_threshold = voice_threshold
        self.silence_duration_s = silence_duration_s
        self.min_speech_s = min_speech_s

        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._running = False

        # VAD state (accessed only from the audio callback thread)
        self._speech_buffer: list[np.ndarray] = []
        self._silence_frames: int = 0
        self._in_speech: bool = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._reset_vad()
            kwargs: dict = {
                "samplerate": self.config.sample_rate_hz,
                "channels": self.config.channels,
                "dtype": "float32",
                "callback": self._audio_callback,
                "blocksize": self._CHUNK_FRAMES,
            }
            if self.config.device is not None:
                kwargs["device"] = self.config.device
            self._stream = sd.InputStream(**kwargs)
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._reset_vad()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_vad(self) -> None:
        self._speech_buffer = []
        self._silence_frames = 0
        self._in_speech = False

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        energy = float(np.mean(np.abs(chunk)))

        if energy >= self.voice_threshold:
            if not self._in_speech:
                self._in_speech = True
                self._silence_frames = 0
                if self.on_voice_start is not None:
                    threading.Thread(target=self.on_voice_start, daemon=True).start()
            self._speech_buffer.append(chunk)
            self._silence_frames = 0
        elif self._in_speech:
            self._speech_buffer.append(chunk)
            self._silence_frames += frames
            silence_s = self._silence_frames / self.config.sample_rate_hz
            if silence_s >= self.silence_duration_s:
                self._flush_utterance()

    def _flush_utterance(self) -> None:
        buf = self._speech_buffer[:]
        self._reset_vad()
        if not buf:
            return
        audio = np.concatenate(buf)
        if audio.size / self.config.sample_rate_hz < self.min_speech_s:
            return  # too short — noise burst, ignore
        threading.Thread(target=self.on_utterance, args=(audio,), daemon=True).start()
