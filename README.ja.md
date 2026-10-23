# Kura

[![English README](https://img.shields.io/badge/README-English-blue)](README.md)
[![Krea 2 LoRA 学習ガイド](https://img.shields.io/badge/Guide-Krea_2_LoRA-blue)](https://comfyui.nomadoor.net/ja/notes/kura-krea2-lora-training/)

Kura は、AI エージェントで LoRA を学習するためのハーネスです。

学習そのものは、[AI-Toolkit](https://github.com/ostris/ai-toolkit)、[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)、[sd-scripts](https://github.com/kohya-ss/sd-scripts) が行います。Kura はそれらを安全に起動し、学習を見守り、記録に残します。

LoRA は一度ではうまく作れません。Kura は、実験を何度も回しやすくするための道具です。

<img width="1905" height="1154" alt="kuramonitor" src="https://github.com/user-attachments/assets/89d09a7e-d5da-4496-86ee-aa14cda30058" />

## 特徴

- 設定、データ、ログ、成果物がすべてファイルに残る
- 同じ手順で、自分の PC でも RunPod でも学習できる
- RunPod の Pod は、回収が終われば消える。PC が落ちても自分で消える（最長 12 時間。`--max-lease` で変えられる）
- 途中から学習を再開できる
- 学習した LoRA を、ComfyUI でそのまま試し描きして比べられる

## 準備

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- AI エージェント（Claude Code、Codex など）
- 自分の PC で学習するなら、Docker と NVIDIA GPU
- RunPod で学習するなら、RunPod の API キー
- 試し描きするなら、[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## セットアップ

```sh
uv tool install git+https://github.com/nomadoor/Kura
```

`kura init` は、今いるフォルダをワークスペースにします。データセット、学習の結果、ダウンロードしたモデル（数十 GB）がここに置かれます。ホームフォルダの直下などで実行せず、Kura 用のフォルダを作り、そこに移動してから実行してください。空き容量に余裕のあるディスクを選んでください。

```sh
mkdir ~/kura && cd ~/kura
kura init
```

必要なら、API キーを登録してください。

```sh
kura secrets set RUNPOD_API_KEY
kura secrets set HF_TOKEN
```

## 使い方

ワークスペースで AI エージェントを起動して、作りたい LoRA を伝えてください。AI が計画を出し、あなたが OK を出すと学習が始まります。

学習の様子は別の端末で見られます。

```sh
kura monitor
```

## 更新

```sh
uv tool upgrade kura
```

## Windows

WSL2 の Ubuntu の中で使ってください。

1. PowerShell（管理者）で `wsl --install` を実行し、再起動する
2. [Docker Desktop](https://docs.docker.com/get-started/get-docker/) を入れ、Settings → Resources → WSL integration で Ubuntu をオンにする
3. 以降は Ubuntu の端末で操作する。ワークスペースは `/mnt/c` ではなく `~/` の下に作る

## もっと詳しく

- [Kura で Krea 2 の LoRA を学習する](https://comfyui.nomadoor.net/ja/notes/kura-krea2-lora-training/)
- [コマンド一覧](docs/commands.md)

## ライセンス

MIT
