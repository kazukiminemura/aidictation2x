# 音声入力アプリ MVP

`docs/mvp_requirements.md` の要件に沿って、ローカル完結のMVPアプリを実装しました。

## 実装機能（MVP）
- 録音開始/停止（同一セッションで文字起こし）
- ローカルASR（Vosk, 日本語モデル）
- 自動編集（句読点、空白整形）
- フィラーワード除去（ON/OFF）
- 話し癖除去（ON/OFF）
- 生テキスト/最終テキスト比較（差分表示）
- 履歴保存（直近N件、初期10件）と直近結果の一時保存

## セットアップ
1. 依存関係をインストール
```bash
pip install -r requirements.txt
```
2. Vosk日本語モデルをダウンロードして配置
- 配置先: `models/vosk-model-ja`
- 例: `models/vosk-model-ja/am/final.mdl` のような構成になること

3. 起動
```bash
python main.py
```

## 設定ファイル
- アプリ設定: `config/app_settings.json`
- フィラーワード/話し癖ルール: `config/text_rules.json`

## 注意
- 外部API呼び出しは行いません（ローカル処理のみ）。
- モデル未配置、マイク未接続、権限拒否時はエラーメッセージを表示します。
