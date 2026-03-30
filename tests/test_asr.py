import sys
from pathlib import Path
import types

import pytest

from src.asr import ASREngine


@pytest.mark.parametrize(
    ("model_id", "dir_name"),
    [
        ("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
        ("openai/whisper-base", "whisper-base"),
    ],
)
def test_convert_model_exports_supported_whisper_models(
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
            Path(save_dir, "processor_config.json").write_text("{}", encoding="utf-8")

    class FakeOVModel:
        def save_pretrained(self, save_dir: Path) -> None:
            Path(save_dir, "openvino_encoder_model.xml").write_text("<xml/>", encoding="utf-8")
            Path(save_dir, "openvino_encoder_model.bin").write_bytes(b"bin")

    def fake_processor_from_pretrained(model_id: str) -> FakeProcessor:
        calls["processor_model_id"] = model_id
        return FakeProcessor()

    def fake_model_from_pretrained(model_id: str, **kwargs) -> FakeOVModel:
        calls["model_model_id"] = model_id
        calls["model_kwargs"] = kwargs
        return FakeOVModel()

    def fake_export_tokenizer(tokenizer: object, output: Path, task: str) -> None:
        calls["tokenizer"] = tokenizer
        calls["tokenizer_task"] = task
        Path(output, "openvino_tokenizer.xml").write_text("<xml/>", encoding="utf-8")
        Path(output, "openvino_tokenizer.bin").write_bytes(b"bin")

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
    assert any("Downloading processor" in message for message in progress_messages)
    assert any("Exporting tokenizer" in message for message in progress_messages)
    assert (tmp_path / dir_name / "openvino_encoder_model.xml").exists()
    assert (tmp_path / dir_name / "openvino_encoder_model.bin").exists()
