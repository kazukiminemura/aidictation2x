import json
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model


class ASREngine:
    def __init__(self, model_dir: Path, sample_rate_hz: int):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Voskモデルが見つかりません: {model_dir}. models配下へ配置してください。"
            )
        self.model = Model(str(model_dir))
        self.sample_rate_hz = sample_rate_hz

    def transcribe(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            return ""

        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        rec = KaldiRecognizer(self.model, self.sample_rate_hz)
        rec.SetWords(True)
        rec.AcceptWaveform(pcm)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()
