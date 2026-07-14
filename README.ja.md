# Kura

[![English README](https://img.shields.io/badge/README-English-blue)](README.md)

Kura は、AIエージェントと一緒に LoRA 学習や生成比較を進めるための、ファイルベースの実験ワークスペースです。

人間は「何をしたいか」と「どのデータを使うか」を決めるだけです。run の作成・実行・監視・比較は AI に任せる、という使い方を想定しています（もちろん手動でCLIを叩いてもいいですが…）。

<img width="1920" height="1080" alt="Kura" src="https://github.com/user-attachments/assets/a23f92f8-460c-40d8-be8f-e42a5ef06f72" />

## Kura とは

Kura は「学習ソフト」そのものではありません。[AI-Toolkit](https://github.com/ostris/ai-toolkit) や [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) といった学習ツールを、**安全に・記録を残しながら・AIから扱いやすく**するための、薄い管理レイヤーです。

学習は Docker（ローカル）または RunPod（リモート）で動かし、設定や結果はすべてただのファイルとして残るので、あとから誰でも中身を確認・再現できます。

ComfyUIを起動しておけば、作成したLoRAと設定したworkflowを使いテスト生成なども可能です。

## はじめに

### 必要なもの

| 必要なもの | 何のため | 入れ方 |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Kura本体を動かすPython環境マネージャ | 下のセットアップ手順0の1行コマンド |
| [Docker](https://docs.docker.com/get-started/get-docker/) | 学習ツールをコンテナで動かす仕組み（仕組みは知らなくても使えます） | Docker Desktop を入れてください |
| NVIDIA GPU | ローカルで学習するなら必要 | RunPodだけ使うなら不要 |
| [RunPod](https://www.runpod.io/) アカウント | クラウドGPUで学習する場合 | API key を取得してenvファイルに記入します |

- **Windows / WSL2 の人**：Docker Desktop の設定で対象ディストリの WSL integration をオンにしてください。
- **Hugging Face token** は、gated/private なモデルを使うときだけ必要です。
- **生成比較（render）** を使うときは、`http://127.0.0.1:8188` で ComfyUI を起動しておきます。

### セットアップ

```sh
# 0. uv を入れる（未導入の人だけ。macOS / Linux / WSL）
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell) の場合: irm https://astral.sh/uv/install.ps1 | iex

# 1. Kura を取得する
git clone https://github.com/nomadoor/Kura.git
cd Kura

# 2. Kura本体と依存をインストール
uv sync

# 3. 作業フォルダと初期設定を作る（datasets/ runs/ workspace.yaml など）
uv run kura init

# 4. 秘密情報を用意する
cp .env.example .env.local
```

`.env.local` を開いて、使う分だけ値を入れてください。**`.env.local` は Git に含まれず、kura コマンド実行時に自動で読み込まれます**。

| 変数 | 入れるとき |
| --- | --- |
| `RUNPOD_API_KEY` | RunPod で学習するなら必要 |
| `HF_TOKEN` | gated/private モデルを使うときだけ |
| `KURA_NTFY_TOPIC` | スマホ/PCに完了通知が欲しいときだけ（任意） |

上記以外に設定が必要な変数はありません（自分の Docker イメージを push する場合だけ、先に `docker login` してください）。

ローカル Docker 実行では、Hugging Face のダウンロードを既定で `cache/huggingface/` に再利用します。このフォルダは Git に入りません。保存場所を変えたい場合は `workspace.yaml` の mount source を変更してください。

## AI と進める流れ

あなた（🧑）が方針を決め、AI（🤖）が手を動かす、という共同作業です。

1. 🧑 `datasets/` にデータセットを置く
2. 🧑 目的を伝える — 例：`このデータセットで Krea 2 のキャラクターLoRAを作りたい`（`rank 16、lr 5e-5…` のように細かく指定してもOK）
3. 🤖 データを調べ、`run.yaml`を書く。RunPodならdraftの`kura run plan`でGPU在庫を確認して即時実行／待機を記録し、その後compileして、設定・前提・resource factsを含む最終planを提示する
4. 🧑 planを一度承認するか、変えたい点を伝える
5. 🤖 planに固定されたローカルDockerまたはRunPod executorで`kura run execute <run-id>`を実行する
6. 🤖 backendや新しい実行経路にsmoke確認が必要な場合は、通常のユーザー操作ではなく開発・診断責務として扱う
7. 🧑 `uv run kura monitor` で様子を見る（🤖 に進捗を報告させてもOK）
8. 🤖 （任意）途中の checkpoint を取り出し、ComfyUI で比較画像を作る
9. 🧑 結果を見て、終了するか追加学習するかを判断（指示すれば 🤖 が実行）

> 💡 データセットの基本的な作り方は、[AI Toolkit で SDXL（Illustrious）LoRA を学習する](https://comfyui.nomadoor.net/ja/notes/ai-toolkit-sdxl-lora-training/) が参考になるかもしれません（SDXL向けですが、考え方は同じです）。

## 様子を見る

`kura monitor` は、実行中/完了済みの run、loss、進捗、GPU/RunPod情報、出力先をまとめて見る**読み取り専用**のTUIです（ここから学習の開始・停止はしません）。

```sh
uv run kura monitor            # 一覧
uv run kura run watch <run-id> # 1本を詳しく
```

## ComfyUI で画像生成（任意）

ComfyUI を起動し、`workflows/` に **API形式**の workflow を置いておけば、学習した LoRA を使った画像生成を AI にやってもらえます。テスト生成、step別・strength別の比較画像づくり、promptset を使ったまとめ生成など、用意した workflow 次第で自由に使えます。

ローカルにGPUが無ければ、`uv run kura render launch <run-id> --executor runpod` で使い捨ての RunPod ComfyUI Pod 上でも生成できます（モデルは自動で Hugging Face から取得、LoRA だけアップロード、完了後に Pod は自動停止）。

> API形式の workflow は ComfyUI の「File → Export (API)」で書き出します（通常のUIエクスポートは `/prompt` が受け付けません）。やり方は [ComfyUI を AIエージェントから使う](https://comfyui.nomadoor.net/ja/data-utilities/ai-agent-api/) を参照してください。

## RunPod の安全設計

RunPod 実行は**使い捨て Pod**が基本です。ローカルで必要な入力だけ送り、学習し、出力を回収したら**自動で止まります**。課金が垂れ流しにならないようになっています。

- 既定では Network Volume を使わない（永続ストレージを残さない／GPUを選びやすい）
- 通常の `kura run execute` は成果物の回収確認後、Podを即時停止する
- draftの`kura run plan`は、RunPod GPU候補の現在の在庫を確認する。選んだGPUが空いていない場合は、空いている代替GPUを選ぶか、compile前に`compute.capacity: {mode: wait, timeout: 6h}`を記録し、最終planでその選択を一度だけ承認する。前景待機中にターミナルを閉じると待機も終了する
- 意図的に確認時間が必要な場合だけ、低水準の `kura run remote <run-id> --hold-for 30m` を使う
- 万一ローカルが落ちても、Pod側に `--max-lease 12h` の課金保険が入っている

## ファイルの置き場所と後片付け

Kura はすべてをワークスペース内のファイルとして置きます。

| 置き場所 | 中身 |
| --- | --- |
| `datasets/<id>/` | あなたのデータセット（画像＋キャプション） |
| `runs/<run-id>/outputs/` | 学習済み LoRA などの成果物 |
| `cache/huggingface/` | ダウンロードしたモデル本体（数十GBになります） |

いずれも Git には入りません。ディスクが気になったら、まず読み取り専用で状況を確認できます：

```sh
uv run kura doctor disk                                           # 何にどれだけ使っているか・空きを確認（読み取り専用）
uv run kura cleanup all                                           # 削除候補をプレビュー（dry-run。--yes で実行）
uv run kura run prune                                             # 古いrunを掃除（--yes で実行）
uv run kura run prune --docker-containers --docker-volumes --yes  # Kura管理の停止コンテナ/volumeも片付ける
```

モデルキャッシュ自体を空けたいときは `cache/huggingface/` を削除（必要になれば自動で再ダウンロードされます）。

## Kura の更新

```sh
git pull
```

更新はこれだけです。依存関係は次の `uv run kura ...` 実行時に自動で揃い、学習用の Docker イメージは Kura の一部として、必要なときに自動で pull され、Kura の更新と一緒に前へ進みます。**ユーザーがイメージをビルド・管理することはありません。**

trainer のバージョンについて2点だけ：

- **AI-Toolkit** はKuraで互換性を確認した本家公式イメージのバージョンで動きます。通常実行では可変の`latest`を追わず、互換性確認後にKura側の固定値を更新します。
- **Musubi Tuner** には公式イメージがないため、Kura が「Kura の Musubi 対応でテスト済みのイメージ」を同梱しています。そもそも Musubi だけ新しくしても新モデルは学習できず（Kura 側の対応が常に必要）、イメージは Kura の更新と一緒に進みます。「自分の Kura は何に対応しているか？」の答えは常に「今の Kura が対応しているもの」の一つだけです。

## もっと詳しく

- [docs/commands.md](docs/commands.md)：コマンド早見表
- [docs/agent-first-cli.md](docs/agent-first-cli.md)：AIが書くもの、CLIが保証するもの、会話状態なしでrunが動く仕組み
- [docs/backend-support.md](docs/backend-support.md)：使用backendの固定バージョンと検証済み経路
- [AGENTS.md](AGENTS.md) と [.claude/skills/](.claude/skills/)：AI向けの常時ルールと作業手順（AIに作業させるときは、まず `AGENTS.md` を読ませる）
- [docs/smoke-test.md](docs/smoke-test.md)：smoke test メモ
- [README.md](README.md)：英語版

## ライセンス

MIT
