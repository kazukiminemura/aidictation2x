# AI Dictation 2x (ASR + OpenVINO LLM)

This app records microphone audio, runs local ASR (Vosk), applies text cleanup rules,
and then post-edits text with a local OpenVINO LLM backend.

## Features
- Push-to-talk recording (`Ctrl+Space`)
- Optional system-wide paste (`Ctrl+Shift+Space`)
- Local ASR (Vosk)
- Rule-based cleanup (fillers, habits, punctuation)
- Personal dictionary (reading -> surface)
- Local OpenVINO LLM post-edit with quality gate and fallback
- History/autosave with LLM metadata

## Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Place a Japanese Vosk model under:
- `models/vosk-model-ja`
- expected file example: `models/vosk-model-ja/am/final.mdl`

3. Ensure OpenVINO model is available as:
- `OpenVINO/Qwen3-8B-int4-cw-ov` (HF/OpenVINO model id)
- or set `llm_model_path` in `config/app_settings.json` to your local OpenVINO model directory

4. Run
```bash
python main.py
```

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
- `llm_device` (`CPU`, etc.)

## Tests
```bash
python -m pytest -q
```
