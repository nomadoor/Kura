# Kura

Kura は、AIエージェントと一緒に LoRA 学習や生成比較を進めるための、ファイルベースの実験ワークスペースです。

手動でCLIを叩いて使うこともできますが、基本的にはデータセットとパラメータだけ用意・指定して、あとはAIに任せる使い方を想定しています。

<img width="1920" height="1080" alt="Kura" src="https://github.com/user-attachments/assets/a23f92f8-460c-40d8-be8f-e42a5ef06f72" />

## 何ができるか

Kuraは「学習ソフトそのもの」ではありません。AI-ToolkitやMusubi Tunerなどのバックエンドを、安全に、記録可能に、AIエージェントから扱いやすくするための実験管理ツールです。

- 学習runを `run.yaml` として残し、あとから再現・確認できる形にする
- AI-Toolkit / Musubi Tuner をDockerまたはRunPodで実行する
- RunPodのPodを一時的に使い、出力をローカルへ回収してから自動停止する
- `kura monitor` で実行中/完了済みrunをTUIで見る
- ComfyUI APIを使って、LoRAのstep別・strength別比較画像を作る
- dataset、workflow、promptset、run出力を分けて管理する
- secret、モデル、checkpoint、生成画像、学習成果物をgitに混ぜない

## 基本の考え方

Kuraでは、ファイルが真実です。

| 場所 | 役割 |
| --- | --- |
| `run.yaml` | 人間またはAIが書いた実験意図 |
| `resolved/` | compile時点で凍結された入力 |
| `realizations/` | launchやreconcileで観測した実行事実 |
| `status.json` | 最新状態の見やすい写し |
| `outputs/` | 回収された成果物 |

データセット本体、モデル、checkpoint、生成画像、RunPodからのdownload、ローカル設定、secretはgit管理しません。

## 前提条件

- Python 3.11+
- `uv`
- ローカル学習やimage buildに使うDocker DesktopまたはDocker Engine
- ローカルでまともに学習するならNVIDIA GPU環境
- RunPodで学習するならRunPodアカウント/API key
- gated/private modelなど、使うモデルによってはHugging Face API token
- render runを使うなら `http://127.0.0.1:8188` で起動しているComfyUI

WSL2で使う場合は、Docker DesktopのWSL integrationを有効にし、Dockerのdisk imageやcache置き場に十分な空き容量を用意してください。

## AIエージェントとの協働の流れ

人間は「何をしたいか」と「どのデータを使うか」を決め、run作成・compile・smoke・RunPod実行・生成比較はAIに任せる、という流れです。

1. `datasets/` にデータセットを置く
2. 目的をAIに伝える  
   例: `このデータセットでキャラクターLoRAを作りたい`
3. パラメータを伝える  
   例: `rank 16、alpha 16、lr 5e-5、batch 2、768px、1500step、100stepごと保存`
4. AIにlocal smokeを作らせる
5. 問題なければローカル(もしくは RunPod) で本番runを回す
6. `kura monitor` で状態を監視
7. 途中checkpointをpullして、ComfyUIで比較画像を作る
8. 良ければ終了、足りなければ追加学習を指示

## Kura monitor

`kura monitor` は、学習や生成runの状態を見るためのTUIです。

実行中のrun、完了したrun、loss、progress、GPU/RunPod情報、出力パスなどをまとめて確認できます。  
ただし、ここから学習を開始・停止する操作系TUIではありません。あくまで読み取り専用の観測画面です。

```sh
uv run kura monitor
uv run kura run watch <run-id>
```

## よく使うコマンド

| コマンド | 目的 |
| --- | --- |
| `uv sync` | 開発環境を用意する |
| `uv run kura init` | workspaceを初期化する |
| `uv run kura doctor docker` | Docker/GPU/cacheまわりを確認する |
| `uv run kura doctor runpod` | RunPod API、Pod、Network Volumeの状態を確認する |
| `uv run kura dataset validate <dataset>` | dataset manifestを確認する |
| `uv run kura run new --experiment <name> --slug <slug>` | 新しいtrain runを作る |
| `uv run kura run compile <run-id>` | `run.yaml` から凍結済み設定を作る |
| `uv run kura run launch <run-id> --executor docker --dry-run` | ローカルDocker実行の事前確認 |
| `uv run kura run launch <run-id> --executor docker` | ローカルDockerで実行する |
| `uv run kura run remote <run-id>` | RunPodで実行し、出力回収後に短時間だけ確認用に残す |
| `uv run kura run pull <run-id> --step <step>` | 実行中のRunPodからcheckpointだけ先に取得する |
| `uv run kura run stop <run-id>` | 対象runのPod/containerを止める |
| `uv run kura run reconcile <run-id>` | 外部状態を読み直してstatusを更新する |
| `uv run kura run prune --dry-run` | 不要run削除の候補を確認する |
| `uv run kura monitor` | TUIでrun一覧と状態を見る |
| `uv run kura run watch <run-id>` | 1本のrunをTUIで詳しく見る |
| `uv run kura render new --slug <slug>` | ComfyUI生成runを作る |
| `uv run kura render compile <run-id>` | workflow/promptsetを凍結する |
| `uv run kura render launch <run-id>` | ComfyUIで生成する |

## RunPodの安全設計

KuraのRunPod実行は、基本的に使い捨てPodです。

1. ローカルでrunをcompileする
2. 必要な入力だけPodへuploadする
3. Pod上で学習する
4. 出力とログをローカルへdownloadする
5. 確認用に短時間だけPodを残す
6. その後、自動でstopする

既定ではNetwork Volumeを使いません。GPUリージョンに縛られにくく、RunPod上に永続ストレージを残さないためです。

完了後、既定では `--hold-for 30m`、つまり30分だけPodを保持し、その後terminateします。この時間にLoRAを確認し、終了するか、追加学習するかを判断してください。

ローカル側が落ちる事故に備えて、Pod側にも `--max-lease 12h` の最後の課金安全弁が入っています。

## Docker image

Kuraのimageには、モデル本体やsecretは入れません。

| backend | 既定のimage |
| --- | --- |
| AI-Toolkit | RunPodではOstris公式の `ostris/aitoolkit:latest` / template を使えます。必要ならlocal imageとして `kura/ai-toolkit:dev` をbuildできます。 |
| Musubi Tuner | remote runは既定で `nomadoor/kura-musubi-tuner:dev`、localは `kura/musubi-tuner:dev` を想定しています。自分のimageを使う場合は `workspace.yaml` で差し替えてください。 |

image名は `workspace.yaml` で設定します。buildは必要なときだけです。

```sh
uv run kura image build ai-toolkit --ref <ref>
uv run kura image build musubi-tuner --ref <ref>
```

## 通知

ローカルに `notify-send` があればデスクトップ通知を使います。  
`.env.local` に `KURA_NTFY_TOPIC` を入れると ntfy 通知も使えます。

```env
KURA_NTFY_TOPIC=long-random-topic-name
```

## AI用ドキュメント

AIエージェント向けの常時ルールは [AGENTS.md](AGENTS.md) にあります。  
細かい作業手順は `.claude/skills/` に分けています。AIに作業させる場合は、まず `AGENTS.md` を読ませてください。

## チェック

```sh
uv run python scripts/check_release.py
```

## 詳細資料

- [README.md](README.md): 英語版README
- [docs/smoke-test.md](docs/smoke-test.md): smoke testメモ

## ライセンス

MIT
