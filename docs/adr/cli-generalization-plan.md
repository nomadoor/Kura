# 実装計画：Kura を「汎用CLI」として詰める

実装担当（AIエージェント可）への作業指示書。チャット履歴に依存しないよう、背景・検証済みの現状・受け入れ条件まで自己完結で記載する。

> この文書は内部向けの作業計画。公開 README からはリンクしない。

## 0. 前提・原則（厳守）

- **判断基準**：ミニマルかつ綺麗に閉じる変更だけ採用。中途半端な状態を残さない。
- **RunPodのみ**：他クラウドの抽象化はしない。executor は RunPod 専用のままでよい。README でも "RunPod remote" と言い切る。
- **配布前提**：当面は git clone 運用が主。pip/wheel のフル対応（package-data 同梱など）は今回やらない。ただし「空ディレクトリで `kura init` してビルドまで破綻しない」ことは担保する。
- **進め方**：各項目は「変更前にコードで現状確認 → 最小変更 → テスト緑」。`uv run python scripts/check_release.py` の**テスト一式が通り続けること**。
- **コミット**：項目ごとに独立コミット。**PR は作らない（コミットまで）。** 明示許可があるまで `gh pr create` しない。
- **render staging について**：以前あった「ComfyUI LoRA staging の既知3点に触るな」という注意は**もう不要**。当該3点（render.py の未使用 `_truthy`、compile が `comfyui` を凍結する順序、`_cleanup_lora_stage` の過剰削除）は既に修正済み。

## 1. フェーズ構成

- **Phase 1 — 機能修正**：現構造のまま小さい差分で入れる。
- **Phase 2 — cli.py 分割**：純粋な抽出（振る舞いゼロ変更）として最後に独立実施。
- 必ず Phase 1 → Phase 2。機能変更とリファクタを混ぜない（差分をレビュー可能に保つため）。

---

## Phase 1：機能修正（この順で実施）

### 1. workspace root の自動探索

- **現状**：`src/kura/cli.py:43` の `_workspace()` が `Path.cwd()` を返すだけ。サブディレクトリから `kura` を叩くと `workspace.yaml` / `.env.local` / `runs/` を見失う。
- **変更**：
  - `_workspace()` を「cwd から親方向へ上がって `workspace.yaml` を持つディレクトリを探し、見つかればそれを返す。見つからなければ cwd を返す」に変更。**探索は決して例外を投げない**（必ず Path を返す）。
  - **エラーは `main()` で雑に出さない。** `kura --help` と `kura init` は workspace 不要。workspace を必要とするコマンド側（または `_require_workspace()` ヘルパ）で、`workspace.yaml` が無いときに明確に落とす：「workspace.yaml が見つかりません。`kura init` するか workspace root で実行してください」。
  - `kura init` は探索結果ではなく **cwd** を使う（まだ workspace.yaml が無いため）。
  - `_load_env_local()` は `main()` 冒頭で走る。探索により**サブディレクトリからでも root の `.env.local` を読む**ようになるのは良い。ただし **workspace 未作成時にエラーにしないこと**（ファイルが無ければ無視する現挙動を維持）。
- **注意**：cwd 依存の既存テストを壊さない。
- **受け入れ条件**：サブディレクトリから `kura monitor` / `kura run status <id>` 等が root の workspace を解決できる。無ワークスペースでは workspace 必要コマンドのみが親切に落ち、`--help` / `init` は動く。テスト一式緑。

### 1b. 「今どの workspace を見ているか」を可視化

- **背景**：サブディレクトリ実行対応を入れると、AI も人間も「どの workspace を見ているか」が分からず事故る。
- **変更**：`kura doctor docker / runpod / comfyui` の出力 JSON に `workspace_root`（解決された絶対パス）を含める。あわせて軽量な `kura doctor workspace`（root・存在する主要サブディレクトリ・`workspace.yaml` 有無を出す read-only）を追加してもよい。
- **受け入れ条件**：各 doctor 出力から現在の workspace root が一目で分かる。

### 2. `kura init` が Dockerfile 参照ファイルを全部生成

- **現状**：`cmd_init`（`src/kura/cli.py` 内、おおよそ 271–288 行）は `workspace.yaml`・両 Dockerfile・musubi patch を生成するが、両 Dockerfile が `COPY` する `docker/ai-toolkit/kura_runpod_object_job.py` を**生成しない**。git clone 時は tracked 済みで問題ないが、init で作った workspace から `kura image build` すると COPY 欠落で失敗し得る。
- **変更**：init が `docker/ai-toolkit/kura_runpod_object_job.py` も生成し、**Dockerfile が `COPY` する全ファイルを init が揃える**。内容は repo 現行ファイルと一致させる。
- **受け入れ条件**：空ディレクトリで `kura init` 後、ai-toolkit / musubi 両イメージの `docker build`（または相当の確認）が COPY 欠落で落ちない。

### 3. `kura doctor comfyui` を追加

- **背景**：ComfyUI LoRA staging を入れたのに doctor は docker / runpod / secrets のみ（`src/kura/cli.py:1657-1660`）。実運用で AI が毎回手で `/object_info` を確認していて詰まる。**機能的詰まりなので boto3 整理より先に入れる。**
- **変更**：`doctor` サブに `comfyui` を追加。既存 doctor と同じ read-only パターンで JSON 報告：
  - endpoint（`workspace.yaml` の `comfyui.endpoint`、既定 `http://127.0.0.1:8188`）への到達可否
  - `/object_info` 取得可否、`LoraLoader` の `lora_name` 候補数（取得できれば）
  - `comfyui.lora_dir` の設定有無、stage dir（`lora_dir/lora_stage_subdir`）の書き込み可否
  - `workspace_root`（1b）
  - **read-only**。生成も変更もしない。
- **受け入れ条件**：`uv run kura doctor comfyui` が ComfyUI 未起動でもクラッシュせず診断 JSON を返す。

### 4. AI 向け workspace 設定リファレンス

- **背景**：`workspace.yaml` の「正」がコード内テンプレートのみ。AI が設定変更時に参照する短い仕様が欲しい。
- **変更**：`docs/workspace-config.md` を新規作成し、キーの簡潔な表：
  - `docker.images.*`、`docker.mounts`（相対パスは workspace 基準で解決／既定 `./cache/huggingface`、`cache/` は gitignore 済み）
  - `comfyui.*`（`endpoint` / `lora_dir` / `lora_stage_subdir` / `lora_stage_mode` / `lora_stage_cleanup`）
  - `runpod.*`（`storage_mode` / `gpu_type_ids` / `cloud_type` / `container_disk_gb` など）
  - **訂正（重要）**：`--hold-for` / `--max-lease` は workspace.yaml ではなく `kura run remote` の **CLI フラグ**。この doc には書かず `docs/commands.md` 側に置く。
  - `AGENTS.md` か関連 skill から 1 行リンク。README 本体は太らせない。
- **受け入れ条件**：表の既定値が実コードと一致。テスト一式緑。

### 5. 未知モデル時の作法を Musubi skill に明記（コード変更なし）

- **背景**：新モデル対応は基本 Kura 更新不要。要素は (1) 重み＝HF 名指定で自動 DL、(2) アーキ対応＝docker image 内のツール次第（Musubi は `MUSUBI_TUNER_REF` を bump して再 build、AI-Toolkit は latest）、(3) Musubi の友好名 bundle マップ（`src/kura/backends.py:341` 付近）のみ Kura 側知識。未登録モデルでも `model_downloads` / `model_paths` 明示で動く。未知モデルのエラーは既に明確（`src/kura/backends.py:321`）。
- **変更**：`.claude/skills/musubi-tuner-backend/SKILL.md`（または項目4の doc）に「未知モデルは `model_downloads`/`model_paths` を明示。bundle 非対応は明示エラーで判断。新アーキは image のツール版を上げて再 build」を 1 ブロック追記。

### 6. boto3 を遅延 import かつ optional 依存へ（依存の綺麗さ。機能詰まりではないので後ろ）

- **現状**：`src/kura/executors.py:19-21` が boto3 / botocore をトップレベル import。利用は object_staging のみ。**`run stage` の object_staging 経路は生きている（`executors.py:523-530`）が、launch は disabled（`executors.py:595-596`）** という半端な状態。
- **変更**：
  - boto3 import を object_staging を実際に使う関数内へ移動（lazy import）。未インストール時は「object_staging は実験的機能。`pip install 'kura[object-staging]'` が必要」と明示エラー。
  - `pyproject.toml`：`boto3` を必須 `dependencies` から `[project.optional-dependencies]` の `object-staging` extra へ移動。
  - **半端状態の解消（決める）**：launch が disabled な以上、stage だけ動くのは中途半端。**object_staging を stage 含めて一貫して experimental 扱いにし**、extra 未導入なら stage 経路も明示エラーにする。あるいは stage 経路も launch と同様に明示 disabled に寄せる。どちらか一方に統一すること（半On を残さない）。
  - **テスト修正必須**：`tests/test_cli.py:1583` が `patch("kura.executors.boto3.client", ...)` を使用。lazy import 後は **patch 先が変わる**（boto3 を import/使用する新しい場所に合わせて patch を更新する）。
  - **触らないもの**：`docker/ai-toolkit/kura_runpod_object_job.py` の boto3 import はコンテナ内で動くスクリプトで、ローカル CLI の env とは別。変更不要。
- **受け入れ条件**：boto3 抜きの環境で全コマンドが import 可能。object_staging 経路（stage/launch）が一貫した挙動（extra 要求 or disabled）。テスト一式緑。

### 7【P3・任意】ミニマルな磨き

- 主要サブコマンドの `add_parser(..., help=...)` に 1 行説明（README を汚さず人間 UX 改善。件数が多いので時間があれば）。
- OS 対応を**README ではなく** `docs/commands.md` か `docs/workspace-config.md` に短く：Linux/WSL2 が主対象、Windows は WSL2+Docker Desktop 推奨、macOS は RunPod 管理と render 管理のみ（ローカル NVIDIA 学習は対象外）。

---

## Phase 2：cli.py モジュール分割（純抽出・振る舞いゼロ変更）

機能修正がすべて入ってから着手する。先にやると機能修正と混ざりレビュー不能になる。

### 原則（厳守）

- **純粋な抽出。ロジック・出力・CLI 仕様を一切変えない。** 関数移動と import 調整のみ。
- **1 モジュール = 1 コミット。** 各コミット後にテスト一式緑、`uv run kura --help` と主要サブコマンド help が従来と同一、doctor 等の JSON 出力が同一であることを確認。
- **import 循環を作らない。** ドメインモジュールは `cli.py` を import しない。共通基盤を最下層に集約し上位が import。
- **綺麗に閉じないものは無理に出さない。** ロジック変更や循環が必要になる断片は `cli.py` に残す。見送った場合は理由をコミットメッセージに明記。

### 既存構造との整合

既に `backends.py`（compile）/ `executors.py`（launch・reconcile・stop・RunPod・docker）/ `render.py` / `monitor.py` / `tui.py` がある。RunPod の実行ロジックは executors.py 側。今回出すのは cli.py 内の**コマンド本体・埋め込みテンプレ・通知・doctor**。

### 目標モジュール（最終形）

1. **`workspace.py`（最下層・基盤）**：`_workspace()`（項目1の探索込み）/ `_require_workspace()` / `_workspace_config()` / `_load_env_local()` / yaml load・dump / `_run_path` / パス系ヘルパ。他モジュールはここを import。
2. **`init_templates.py`**：埋め込み巨大文字列（workspace.yaml / 両 Dockerfile / musubi patch / kura_runpod_object_job.py）＋ `cmd_init`。
3. **`notifications.py`**：ntfy / notify-send 一式。
4. **`doctor.py`**：`cmd_doctor_docker` / `runpod` / `secrets` / `comfyui` / `workspace`。
5. **（任意・最後）`run_commands.py`**：`cmd_run_remote` と hold-for/max-lease 制御・pull/stop/reconcile/stage/upload/download のオーケストレーション。**最難関**。綺麗に出せない/循環が出るなら分割せず cli.py に残す。
6. **`cli.py`（最終的に薄く）**：argparse 配線と各 `cmd_*` への dispatch のみ。

### 実施順

`workspace.py` → `init_templates.py` → `notifications.py` → `doctor.py` →（可能なら）`run_commands.py`。基盤から上へ。

### 受け入れ条件

- 全工程を通じて `kura` の外部挙動（help 文・サブコマンド・JSON 出力・終了コード）が**不変**。
- テスト一式緑、`check_readme_cli_sync.py` 緑。
- 各モジュールが単体 import 可能、循環なし。
- `cli.py` が大幅減（目安：argparse＋dispatch 中心の数百行台）。
- `run_commands.py` を見送った場合は理由をコミットメッセージに明記。

---

## 実装順（全体）

Phase 1：1（＋1b）→ 2 → 3 → 4 → 5 → 6 →（任意 7）
Phase 2：workspace.py → init_templates.py → notifications.py → doctor.py →（可能なら）run_commands.py

各コミット後に `uv run python scripts/check_release.py` と関連 `kura ... --help` を実行。**PR は作らない。**
