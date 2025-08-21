# Spunky プロジェクト
# まだ完成してません♡
ps.動くようにはなった
本コードのコメントはAIによるものです
## 1. Spunkyって何？

一言でいうと、「LLMにちゃんと"考え"させる」ための実験的なアーキテクチャ、あるいはその実装です。

昨今のLLMは非常に流暢でクリエイティブな文章を生成できますが、時々、平気で嘘をついたり（幻覚）、会話の文脈を無視したり、論理的な一貫性がなかったりします。これは、LLMが巨大な確率モデルであり、厳密な意味で「思考」しているわけではないからです。

Spunkyは、この問題に「役割分担」というアプローチで挑みます。

- **Spunky-Core (Pythonベースの実行エンジン)**
  - 「何を」「どの順番で」「なぜ」話すか、という論理的な思考、計画、記憶検索、ツール利用を担当します。いわば「思考の運転手」です。
  - 思考プロセスは透明性が高く、制御可能です。

- **大規模言語モデル (LLM / このプロジェクトではQwen)**
  - Spunky-Coreが考えた「応答の設計図（Content Plan）」を元に、人間らしい自然な文章を生成することに特化します。いわば「表現力がすごい役者」です。

このハイブリッドな「共同運転（Co-Driver）モデル」によって、LLMの表現力を損なうことなく、思考の安定性と制御可能性を両立させることを目指しています。

## 0. プロジェクト構成（ファイル構成）

```
Spunky/
├─ docs/
│  └─ specification.md            # 仕様や設計ノート
├─ spunky/                         # アプリ本体（Pythonパッケージ）
│  ├─ __init__.py
│  ├─ config.py                    # 共通設定/環境変数の読み込み
│  ├─ main.py                      # CLI/簡易エントリ（必要なら使用）
│  ├─ server.py                    # FastAPI サーバー（エントリ）
│  ├─ utils.py                     # 汎用ユーティリティ
│  ├─ chat/                        # 会話生成（NLG/NLU/対話管理）
│  │  ├─ __init__.py
│  │  ├─ character.yml             # キャラクター/ペルソナ定義
│  │  ├─ dialog_manager.py         # 対話状態/履歴の整理
│  │  ├─ feedback.py               # 応答の自己改善フィードバック
│  │  ├─ main_chat_loop.py         # チャット制御のループ
│  │  ├─ nlg.py                    # 応答文生成（Transformers/llama.cpp対応）
│  │  ├─ nlu.py                    # 軽量NLU（意図/感情の推定）
│  │  └─ self_improve.py           # 自己改善ロジック
│  ├─ knowledge/                   # RAG/知識検索
│  │  ├─ __init__.py
│  │  └─ retriever.py              # 検索/埋め込み/FAISS
│  ├─ memory/                      # 記憶/学習/状態管理
│  │  ├─ __init__.py
│  │  ├─ learning.py               # 継続学習まわり
│  │  ├─ state_manager.py          # 会話/ユーザ状態の管理
│  │  ├─ svm_emotion_model.pkl     # 感情SVMモデル
│  │  └─ svm_intent_model.pkl      # 意図SVMモデル
│  ├─ scripts/
│  │  ├─ evaluate.py               # 評価スクリプト
│  │  └─ train.py                  # 学習スクリプト
│  ├─ data/                        # データ置き場（空フォルダ）
│  ├─ evaluation/                  # 評価用成果物/ログ
│  └─ think/                       # 思考（Reasoning）中枢
│     ├─ __init__.py
│     ├─ approach_formation.py     # アプローチ策定
│     ├─ core.py                   # ThinkCore本体（仮説→実行）
│     ├─ evaluation.py             # 仮説評価
│     ├─ fam.py                    # Failure-Aware Mechanism等
│     ├─ finalization.py           # 結果の取りまとめ
│     ├─ reason_analysis.py        # 推論過程の分析
│     └─ strategy.py               # 戦略選択
├─ howto.md                         # 簡単な使い方メモ
├─ readme.md                        # このファイル
├─ requirements.txt                 # 依存パッケージ
├─ shell.py / shell.ps1             # ローカル実行補助スクリプト
├─ tellme.md / tellyou.log          # 実験ログ/メモ
```

要点（主要コンポーネント）
- server.py: FastAPIエントリ。/chat, /clear_history などのエンドポイント、応答の最終サニタイズ、GGUF設定の受け渡しを担当。
- chat/nlg.py: 応答生成器。Transformers+LoRA と llama.cpp(GGUF) を自動切替。/think と /no-think のモード、空応答抑制の後処理、雑談高速応答を実装。
- think/core.py: 推論中枢。仮説生成→評価→実行でツール/RAG/記憶を統合し、Content Plan（応答設計図）を作る。
- knowledge/retriever.py: RAG検索。FAISS/埋め込みの管理。
- memory/state_manager.py: 会話履歴やユーザ状態の管理。
- chat/nlu.py: 意図/感情推定（RoBERTaベース SVM 付随）。
- chat/character.yml: キャラクター/口調などのペルソナ設定。

補足（GGUF/llama.cpp の設定）
- 環境変数でバックエンドを制御できます：
    - SPUNKY_GGUF_MODEL: GGUFモデルのパス（指定時はllama.cppを優先）
    - SPUNKY_LLAMA_N_CTX: コンテキスト長（例: 8192）
    - SPUNKY_LLAMA_N_GPU_LAYERS: GPUへオフロードする層数（CUDAビルド時）
    - SPUNKY_LLAMA_N_THREADS: CPUスレッド数
    - SPUNKY_LLAMA_MAIN_GPU: メインGPUインデックス（例: 0）
    ※ CUDA対応のllama-cpp-pythonを使用している場合のみGPUオフロードが有効になります。

リモート推論（Ollama / llama.cpp server）
- 起動済みのリモートがあれば自動で優先使用します（優先度: Ollama > llama.cpp server > ローカルGGUF > Transformers+LoRA）。
- 環境変数（任意）
    - SPUNKY_OLLAMA_BASE_URL: OllamaのURL（既定: http://127.0.0.1:11434）
    - SPUNKY_OLLAMA_MODEL: 使うモデル名（未指定なら qwen3/qwen2 を優先自動選択）
    - SPUNKY_LLAMA_SERVER_URL: llama.cpp server(OpenAI互換)のURL（既定: http://127.0.0.1:8080）
    - SPUNKY_LLAMA_SERVER_MODEL: 使うモデルID（未指定なら qwen3/qwen2 を優先自動選択）

注意（Qwen3 GGUF）
- 一部のllama.cppビルドではQwen3アーキテクチャ未対応です。その場合は下記いずれかを推奨：
    1) Qwen2.x / Qwen2.5 のGGUFを使う
    2) Qwen3に対応した新しいllama.cpp（CUDA対応）を用意する

## 2. どうやって動いてるの？ (ワークフロー)

ユーザーからリクエストが来ると、Spunkyの中ではこんな処理が走っています。

1.  **受付 (`server.py`)**
    - FastAPIで実装されたAPIサーバーが、リクエストを受け取ります。

2.  **司令塔 (`orchestrator.py`)**
    - 受け取ったリクエストを、まず交通整理員に渡します。

3.  **交通整理 (`router.py`)**
    - リクエストの内容をざっと見て、「これは雑談だな」「これは天気についての質問だな」「ちゃんと考えるべき問題だな」というように、行き先を決めます。キーワードや、軽量なNLUモデル（RoBERTa）がこの判断を担います。

4.  **頭脳 (`think/core.py`)**
    - 「ちゃんと考えるべき」と判断された場合、思考の中枢であるThinkCoreが動き出します。
    - ThinkCoreは、どの道具を使えばいいか仮説を立て、評価し、最適なものを選択します。
    - **道具箱 (`think/tools/`)**: 天気API、Pythonインタプリタ（計算機）などの専門的なツールがここに入っています。
    - **記憶 (`knowledge/`, `memory/`)**: 事前に学習した知識（Wikipediaなど）や、過去の会話の記憶も、ここで参照されます。
    - 最終的に、ThinkCoreは「何を話すべきか」をまとめた**応答設計図 (Content Plan)** を作成します。

5.  **役者 (`chat/nlg.py`)**
    - 思考結果である「応答設計図」を、表現のプロであるLLM（Qwen-LoRA）に渡します。
    - LLMは、設計図とキャラクター設定（妹口調など）に基づいて、最終的な応答のセリフを生成します。

6.  **応答**
    - 生成されたセリフが、ユーザーに返されます。

## 3. よくある質問 (FAQ)

**Q: このプロジェクトのゴールは？**

A: LLMの「それっぽい嘘」を減らし、もっと信頼・制御できる形でAIエージェントを動かすためのアーキテクチャを模索することです。Spunky-Coreが思考の主導権を握ることで、なぜその応答になったのか、後から検証しやすくなります。

**Q: なぜハイブリッド構成なの？LLMだけで全部できない？**

A: LLM単体だと、思考のプロセスがブラックボックスになりがちで、制御が難しいからです。得意なことは得意なやつに任せる「役割分担」が、今のところ堅実なアプローチだと考えています。

**Q: なぜCPUのみが使われているか？**

A: 基本的には、特別なGPUマシンがなくても動かせることを目指しています。Spunky-Coreの思考部分はPythonなのでCPUで十分動きます。LLM（Qwen）や各種AIモデルの推論は、GPUがあればもちろん高速になりますが、CPUでも動作する設定になっています。環境への依存をなるべく減らしたい、という意図です。

**Q: 新しいツールを追加するには？**

A: `spunky/think/tools/` に新しいツールのPythonファイルを追加し、`think/core.py` の仮説生成と実行の部分に、そのツールを呼び出すロジックを追加します。`router.py` に、そのツールを呼び出すためのキーワード判定を追加するのも良いでしょう。

**Q: キャラクターの設定はどこで変えるの？**

A: `spunky/chat/character.yml` で、ペルソナや応答テンプレートを定義しています。LLMに与えるSystem Promptも `spunky/chat/nlg.py` 内で定義されているので、そこを調整することでキャラクター性を変更できます。

## 4. 動かし方

**1. サーバーの起動**

```bash
python spunky/server.py
```

**2. クライアントの起動**

別のターミナルを開いて、以下を実行します。

```bash
streamlit run client.py
```

ブラウザで `http://localhost:8501` が開かれ、チャット画面が表示されます。
