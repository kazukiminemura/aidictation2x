import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.asr import ASREngine, _QwenASREngine, _dedupe_repeated_text


@pytest.mark.parametrize(
    ("model_id", "dir_name"),
    [
        ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B"),
        ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR-0.6B"),
    ],
)
def test_convert_model_downloads_supported_qwen_model(
    monkeypatch,
    tmp_path: Path,
    model_id: str,
    dir_name: str,
) -> None:
    progress_messages: list[str] = []

    def fake_snapshot_download(**kwargs) -> str:
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    hub_module = types.ModuleType("huggingface_hub")
    hub_module.snapshot_download = fake_snapshot_download
    hub_utils_module = types.ModuleType("huggingface_hub.utils")
    hub_utils_module.disable_progress_bars = lambda: None

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", hub_utils_module)

    engine = ASREngine(sample_rate_hz=16000, model_id=model_id, models_root_dir=tmp_path)
    model_path = engine.convert_model(progress_callback=progress_messages.append)

    assert model_path == str(tmp_path / dir_name)
    assert any("Downloading ASR files" in message for message in progress_messages)
    assert (tmp_path / dir_name / "config.json").exists()
    assert (tmp_path / dir_name / "tokenizer_config.json").exists()


@pytest.mark.parametrize(
    ("model_id", "dir_name"),
    [
        ("openai/whisper-base", "whisper-base"),
        ("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
    ],
)
def test_convert_model_exports_supported_whisper_ir_models(
    monkeypatch,
    tmp_path: Path,
    model_id: str,
    dir_name: str,
) -> None:
    calls: dict[str, object] = {}
    progress_messages: list[str] = []

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = object()

        def save_pretrained(self, save_dir: Path) -> None:
            (Path(save_dir) / "processor_config.json").write_text("{}", encoding="utf-8")

    class FakeOVModel:
        def save_pretrained(self, save_dir: Path) -> None:
            (Path(save_dir) / "openvino_encoder_model.xml").write_text("<xml/>", encoding="utf-8")
            (Path(save_dir) / "openvino_encoder_model.bin").write_bytes(b"bin")

    def fake_processor_from_pretrained(repo_id: str) -> FakeProcessor:
        calls["processor_model_id"] = repo_id
        return FakeProcessor()

    def fake_model_from_pretrained(repo_id: str, **kwargs) -> FakeOVModel:
        calls["model_model_id"] = repo_id
        calls["model_kwargs"] = kwargs
        return FakeOVModel()

    def fake_export_tokenizer(tokenizer: object, output: Path, task: str) -> None:
        calls["tokenizer"] = tokenizer
        calls["tokenizer_task"] = task
        (Path(output) / "openvino_tokenizer.xml").write_text("<xml/>", encoding="utf-8")
        (Path(output) / "openvino_tokenizer.bin").write_bytes(b"bin")

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = types.SimpleNamespace(from_pretrained=fake_processor_from_pretrained)
    optimum_intel_module = types.ModuleType("optimum.intel")
    optimum_intel_module.OVModelForSpeechSeq2Seq = types.SimpleNamespace(from_pretrained=fake_model_from_pretrained)
    optimum_convert_module = types.ModuleType("optimum.exporters.openvino.convert")
    optimum_convert_module.export_tokenizer = fake_export_tokenizer

    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_intel_module)
    monkeypatch.setitem(sys.modules, "optimum.exporters", types.ModuleType("optimum.exporters"))
    monkeypatch.setitem(sys.modules, "optimum.exporters.openvino", types.ModuleType("optimum.exporters.openvino"))
    monkeypatch.setitem(sys.modules, "optimum.exporters.openvino.convert", optimum_convert_module)

    engine = ASREngine(sample_rate_hz=16000, model_id=model_id, models_root_dir=tmp_path)
    model_path = engine.convert_model(progress_callback=progress_messages.append)

    assert model_path == str(tmp_path / dir_name)
    assert calls["processor_model_id"] == model_id
    assert calls["model_model_id"] == model_id
    assert calls["model_kwargs"] == {"export": True, "compile": False}
    assert calls["tokenizer_task"] == "automatic-speech-recognition"
    assert any("Exporting" in message for message in progress_messages)
    assert (tmp_path / dir_name / "openvino_encoder_model.xml").exists()
    assert (tmp_path / dir_name / "openvino_tokenizer.xml").exists()


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("same sentence. same sentence.", "same sentence."),
        ("hello hello", "hello hello"),
        ("konnichiwa konnichiwa", "konnichiwa konnichiwa"),
        ("test. test. next.", "test. next."),
    ],
)
def test_dedupe_repeated_text(raw_text: str, expected: str) -> None:
    assert _dedupe_repeated_text(raw_text) == expected


def test_qwen_engine_transcribe_passes_audio_tuple(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeTorch:
        bfloat16 = "bfloat16"
        float32 = "float32"

    class FakeModel:
        def transcribe(self, *, audio: object, language: str | None) -> list[object]:
            calls["audio"] = audio
            calls["language"] = language
            return [types.SimpleNamespace(text=" hello ")]

    class FakeQwen3ASRModel:
        @staticmethod
        def from_pretrained(*args, **kwargs) -> FakeModel:  # noqa: ANN002, ANN003
            return FakeModel()

    qwen_module = types.ModuleType("qwen_asr")
    qwen_module.Qwen3ASRModel = FakeQwen3ASRModel
    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = FakeTorch.bfloat16
    torch_module.float32 = FakeTorch.float32

    monkeypatch.setitem(sys.modules, "qwen_asr", qwen_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    engine = _QwenASREngine(model_source="dummy", device="cpu", language="ja")
    text = engine.transcribe(np.array([0.1, -0.2], dtype=np.float32), 22050)

    assert text == "hello"
    assert isinstance(calls["audio"], tuple)
    assert calls["audio"][1] == 22050
    assert np.array_equal(calls["audio"][0], np.array([0.1, -0.2], dtype=np.float32))
