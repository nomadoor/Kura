# Kura

[![English README](https://img.shields.io/badge/README-English-blue)](README.md)
[![Krea 2 LoRA 学習ガイド](https://img.shields.io/badge/Guide-Krea_2_LoRA-blue)](https://comfyui.nomadoor.net/ja/notes/kura-krea2-lora-training/)

Kura は、AI エージェントと一緒に LoRA を学習するためのワークスペースです。

あなたが決めるのは「何を作りたいか」と「どのデータを使うか」だけです。設定を考え、学習を回し、ComfyUI で試し描きして結果を比べる作業は、AI に任せられます。

<img width="1905" height="1154" alt="kuramonitor" src="https://github.com/user-attachments/assets/89d09a7e-d5da-4496-86ee-aa14cda30058" />

## Kura がやること

Kura 自体は学習ソフトではありません。[AI-Toolkit](https://github.com/ostris/ai-toolkit)、[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)、[sd-scripts](https://github.com/kohya-ss/sd-scripts) という学習ツールを、安全に、記録を残しながら、AI から扱えるようにする管理役です。

- **渡したデータだけで学習する**：データセットに列挙したファイルだけが学習に使われます。フォルダに紛れ込んだ古いファイルや下書きの画像が、知らないうちに混ざることはありません。
- **動く設定から始められる**：モデルごとに必要な細かい設定は、Kura が学習ツール本来の推奨値で埋めます。あなたや AI が指定するのは、変えたいところだけです。
- **学習から試し描きまで一続きで**：学習した LoRA を、あなたの ComfyUI ワークフローでそのまま試し描きし、条件を変えて比べられます。
- **自分の PC でもクラウドでも**：自分の GPU でも、[RunPod](https://www.runpod.io/) のクラウド GPU でも、同じ手順で学習できます。
- **全部ファイルに残る**：設定、使ったデータ、ログ、成果物は、すべてワークスペースにただのファイルとして残ります。あとから中身を確かめたり、同じ学習をやり直したりできます。

## 必要なもの

| 必要なもの | 何のため | いつ必要か |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Kura を動かす | 常に（下のセットアップで入ります） |
| Docker | 学習ツールを動かす | 自分の PC で学習するとき。Windows と Mac は [Docker Desktop](https://docs.docker.com/get-started/get-docker/)、Linux は Docker Engine |
| NVIDIA GPU | 学習する | 自分の PC で学習するとき |
| [RunPod](https://www.runpod.io/) アカウント | クラウド GPU で学習する | RunPod を使うとき |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | 試し描きする | 試し描きするとき |

RunPod だけで学習する場合は、手元に GPU は要りません。Mac には NVIDIA GPU がないので、学習は RunPod で行います。

## Windows で使う場合

Windows では、Kura を WSL2（Windows の中で動く Ubuntu）の中で使います。Windows で直接 Kura を動かすことはできません。

1. **WSL2 と Ubuntu を入れる**：PowerShell を管理者として開き、`wsl --install` を実行して PC を再起動します。再起動後に開く Ubuntu で、ユーザー名とパスワードを決めてください。すでに WSL の Ubuntu を使っている人は不要です。
2. **Docker Desktop を入れる**：[Docker Desktop](https://docs.docker.com/get-started/get-docker/) をインストールして起動し、Settings → Resources → WSL integration で Ubuntu をオンにします。
3. **NVIDIA のドライバを最新にする**：自分の PC で学習する場合は、Windows 側に最新の NVIDIA ドライバを入れておきます。

以降のセットアップと操作は、すべて Ubuntu の端末で行います。Kura のフォルダは `/mnt/c` の下ではなく、Ubuntu のホーム（`~/`）の下に置いてください（`/mnt/c` の下だと学習時の読み込みが遅くなります）。

## セットアップ

```sh
# 1. uv を入れる（入っていない人だけ）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Kura を取ってきて、準備する
git clone https://github.com/nomadoor/Kura.git
cd Kura
uv sync          # Kura と必要なものを入れる
uv run kura init # 作業用のフォルダと初期設定を作る

# 3. 秘密の設定ファイルを作る
cp .env.example .env.local
```

`.env.local` には、使うものだけ書き込みます。このファイルは Git に入らず、Kura が自動で読み込みます。

| 変数 | 必要なとき |
| --- | --- |
| `RUNPOD_API_KEY` | RunPod で学習する |
| `HF_TOKEN` | 利用申請が必要なモデル（FLUX.1-dev など）を使う |
| `KURA_NTFY_TOPIC` | 学習の完了をスマホや PC に通知したい（任意） |

準備ができたかは、次で確かめられます。

```sh
uv run kura doctor docker   # Docker と GPU が使えるか（自分の PC で学習する人）
uv run kura doctor runpod   # RunPod の設定が正しいか（RunPod を使う人）
```

## 使い方

### 流れ

AI エージェント（Claude Code、Codex など）を Kura のフォルダで起動し、話しかけて進めます。AI は、Kura のルール（`AGENTS.md`）を読んでから作業します。

1. 🧑 `datasets/<名前>/` に、画像とキャプション（画像と同じ名前の `.txt`）を置く
2. 🧑 やりたいことを伝える。例：「このデータセットで Krea 2 のキャラクター LoRA を作りたい」。rank や学習率などを細かく指定してもかまいません
3. 🤖 データを確認して、学習に使うファイルの一覧（`dataset.yaml` と `items.jsonl`）と学習の設定を作る
4. 🤖 設定、学習する場所（自分の PC か RunPod か）、かかる時間や費用の見込みをまとめた「計画（plan）」を見せる
5. 🧑 計画を確かめて OK を出す。変えたい点があれば伝える
6. 🤖 学習を実行し、終わったら結果を報告する
7. 🤖 ComfyUI で試し描きし、途中で保存した LoRA や条件を変えて並べて比べる
8. 🧑 結果を見て、終わりにするか、続けて学習するかを決める（続きからの学習にも対応しています）

> 💡 データセットの作り方の考え方は、[AI Toolkit で SDXL（Illustrious）LoRA を学習する](https://comfyui.nomadoor.net/ja/notes/ai-toolkit-sdxl-lora-training/) が参考になります（SDXL 向けですが、考え方は共通です）。

### 学習する場所：自分の PC か RunPod か

- **自分の PC**：Docker の中で学習します。モデルは一度ダウンロードすれば、次からは使い回されます。
- **RunPod**：学習に必要なファイルだけを送り、学習して、成果物を回収したら **Pod は自動で止まります**。使いたい GPU が空いていないときは、別の GPU にするか空くまで待つかを計画の段階で選べます（待っている間は課金されません）。万一あなたの PC が落ちても、学習が終わって成果物を回収できないまま一定時間（2 時間と学習時間の長いほう）たつと、Pod は自分で止まります。さらに最大稼働時間（`--max-lease`、既定 12 時間）を過ぎた Pod も自分で止まります。

どちらを使うかは、GPU の大きさや費用を見て AI が提案し、計画の中であなたが決めます。

### ComfyUI で試し描き

試し描きには、ComfyUI と、`workflows/` に置いた **API 形式**のワークフローを使います。

- ComfyUI を `http://127.0.0.1:8188` で起動しておきます。
- ワークフローは、ComfyUI の「File → Export (API)」で書き出して `workflows/` に置きます。詳しくは [ComfyUI を AI エージェントから使う](https://comfyui.nomadoor.net/ja/data-utilities/ai-agent-api/) を参照してください。
- 手元に GPU がなければ、RunPod 上の使い捨ての ComfyUI でも描けます。

### 様子を見る

学習の様子は、別の端末から見られます。見るための画面なので、ここから学習を始めたり止めたりはしません。

```sh
uv run kura monitor             # すべての学習を一覧
uv run kura run watch <run-id>  # 1 つを詳しく
```

## ファイルの置き場所と片付け

| 場所 | 中身 |
| --- | --- |
| `datasets/<名前>/` | あなたのデータセット |
| `runs/<run-id>/outputs/` | 学習した LoRA |
| `artifacts/training-state/` | 続きから学習するための保存データ |
| `cache/huggingface/` | ダウンロードしたモデル（数十 GB になります） |

どれも Git には入りません。ディスクの空きが気になったら、まず確認だけできます。

```sh
uv run kura doctor disk   # 何にどれだけ使っているか（見るだけ）
uv run kura cleanup all   # 消せるものの一覧（--yes を付けると削除）
uv run kura run prune     # 古い学習の一覧（--yes を付けると削除）
```

モデルのキャッシュを空けたいときは、`cache/huggingface/` を削除してください。必要になれば、自動でまたダウンロードされます。

## 対応しているモデル

主な対応は次のとおりです。どこまで実際に学習を通して確かめたかの詳細は、[対応表](docs/backend-support.md)にあります。

| 学習ツール | 主なモデル |
| --- | --- |
| sd-scripts | SD 1.5、SDXL、FLUX.1、Anima（LoRA / ControlNet-LLLite） |
| Musubi Tuner | Wan、FLUX.2、Krea 2、Qwen-Image、Z-Image、FLUX.1 Kontext、HiDream-O1、Ideogram 4、HunyuanVideo、FramePack、Kandinsky 5、MiniMax-H3 など |
| AI-Toolkit | SD 1.5、SDXL、FLUX.1 / Kontext / Flex.2、Chroma、Qwen-Image、FLUX.2、Krea 2、Z-Image、HiDream、Anima、MiniMax-H3 など |

ここでの「対応」は、学習が動き、成果物が保存されるところまで確かめたという意味です。どんなデータや設定でも良い LoRA ができる、という保証ではありません。

## Kura の更新

```sh
git pull
```

これだけです。必要なものは、次に `uv run kura ...` を実行したときに自動でそろいます。学習ツールの Docker イメージも Kura が管理していて、動作を確かめたバージョンに固定されています。自分でビルドしたり、バージョンを選んだりする必要はありません。

## もっと詳しく

- [Kura で Krea 2 の LoRA を学習する](https://comfyui.nomadoor.net/ja/notes/kura-krea2-lora-training/)：データの準備から ComfyUI での比較までの実例
- [docs/commands.md](docs/commands.md)：コマンド一覧
- [docs/backend-support.md](docs/backend-support.md)：対応表と、確かめた範囲
- [docs/agent-first-cli.md](docs/agent-first-cli.md)：AI が書くものと、Kura が保証するもの
- [AGENTS.md](AGENTS.md)：AI エージェント向けのルール

## ライセンス

MIT
