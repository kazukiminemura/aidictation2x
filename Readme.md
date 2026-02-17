# AI Dictation 2x (ASR + OpenVINO LLM)

This app records microphone audio, runs local ASR (Vosk / Whisper), applies text cleanup rules,
and then post-edits text with a local OpenVINO LLM backend.

## Features
- Push-to-talk recording (`Ctrl+Space`)
- Optional system-wide paste (`Ctrl+Shift+Space`)
- Local ASR (Vosk / Whisper)
- Rule-based cleanup (fillers, habits, punctuation)
- Personal dictionary (reading -> surface)
- Local OpenVINO LLM post-edit with quality gate and fallback
- Optional conversion to business email format
- History/autosave with LLM metadata

## Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Place a Japanese Vosk model under:
- `models/vosk-model-ja`
- expected file example: `models/vosk-model-ja/am/final.mdl`

3. OpenVINO model handling:
- Default `llm_model_path` is `OpenVINO/Qwen3-8B-int4-cw-ov`
- Use the **Download LLM Model** button in the app to download from Hugging Face
- Download target defaults to `models/openvino`
- You can also set `llm_model_path` to a local OpenVINO model directory

4. Run
```bash
python main.py
```

## Windows installer build
1. Install build tools
```powershell
pip install pyinstaller
```
- Install Inno Setup 6 (optional, needed only for `.exe` installer generation)
- `iscc.exe` must be available on PATH

2. Build app + installer
```powershell
powershell -ExecutionPolicy Bypass -File .\build\windows\build.ps1 -Version 0.1.0
```

3. Outputs
- App bundle: `dist/staging/AIDictation2x-<version>/AIDictation2x/`
- Installer: `dist/installer/AIDictation2x-Setup-<version>.exe` (when Inno Setup is installed)

4. Runtime data location (installed app)
- `%LOCALAPPDATA%\AIDictation2x\`
- Config/data are copied on first launch:
  - `config/app_settings.json`
  - `config/text_rules.json`
  - `config/personal_dictionary.json`
  - `data/history.json`, `data/last_session.json`

## LLM settings
`config/app_settings.json` keys:
- `llm_enabled`
- `llm_model_path`
- `llm_strength` (`weak` | `medium` | `strong`)
- `llm_max_input_chars`
- `llm_max_change_ratio`
- `llm_domain_hint`
- `llm_timeout_ms`
- `llm_blocked_patterns`
- `llm_device` (`GPU` recommended, `CPU` fallback)
- `llm_auto_download` (`true` | `false`, default `false`)
- `llm_download_dir` (local cache/download folder)

## ASR settings
`config/app_settings.json` keys:
- `asr_backend` (`vosk` | `whisper`)
- `vosk_model_dir`
- `whisper_model_name` (default: `OpenVINO/whisper-large-v3-int8-ov`)
- `whisper_device` (`auto` | `cpu` | `cuda`)
- `whisper_download_dir` (pre-download destination for Whisper models)

You can also switch ASR backend/model from `Properties...` in the app.
Use `Download ASR Model (Whisper)` in `Properties...` to pre-download before recording.
Whisper ASR uses OpenVINO (`openvino_genai.WhisperPipeline`) and expects an OpenVINO-converted model directory.

## UI options
- Right-click anywhere in the main window and open `Properties...`
- You can toggle:
  - Auto edit / Remove fillers / Remove habits
  - Enable LLM correction
  - System-wide input
  - Convert to business email

## Tests
```bash
python -m pytest -q
```
