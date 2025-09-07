# Spunky プロジェクト
本コードのコメントはAIによるものです

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">オタ知識&amp;妹口調特化ローカルLLM完成ed<br>Geminiより...? <a href="https://t.co/hf4l9vKrW6">pic.twitter.com/hf4l9vKrW6</a></p>&mdash; Rinta (@Rinta_LAL) <a href="https://twitter.com/Rinta_LAL/status/1962709331193012594?ref_src=twsrc%5Etfw">September 2, 2025</a></blockquote> 
<img width="1083" height="770" alt="スクリーンショット 2025-09-02 114442" src="https://github.com/user-attachments/assets/acbaf7a2-e509-40e5-8552-dfe8a3f07e4c" />


## 1. Spunkyって何？

一言でいうと、「LLMにちゃんと"考え"させる」ための実験的なアーキテクチャ、あるいはその実装です。

昨今のLLMは非常に流暢でクリエイティブな文章を生成できますが、時々、平気で嘘をついたり（幻覚）、会話の文脈を無視したり、論理的な一貫性がなかったりします。これは、LLMが巨大な確率モデルであり、厳密な意味で「思考」しているわけではないからです。

Spunkyは、この問題に「役割分担」というアプローチで挑みます。

- **Spunky-Core (Pythonベースの実行エンジン)**
  - 「何を」「どの順番で」「なぜ」話すか、という論理的な思考、計画、記憶検索、ツール利用を担当します。いわば「思考の運転手」です。
  - 思考プロセスは透明性が高く、制御可能です。

- **大規模言語モデル (LLM / このプロジェクトではQwen)**
  - Spunky-Coreが考えた「応答の設計図（Content Plan）」を元に、人間らしい自然な文章を生成することに特化します。いわば「表現力がすごい役者」です。

- **豊富なRAG知識**
   - ~~[ニコニコ大百科のデータ](https://www.nii.ac.jp/dsc/idr/nico/)（提供: 株式会社ドワンゴおよび国立情
    報学研究所）をすべてインデックスしており、 オタク向け知識を追加しています~~
　 -　現在は、ライセンスの返信待ちのため一時的に適当な勉強テキストを入れています

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
- server.py: FastAPIエントリ。/chat エンドポイント、応答の最終サニタイズ、GGUF設定の受け渡しを担当（サーバーはステートレス）。
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

## 5. サーバーAPIドキュメント（簡易）

エンドポイント一覧
- GET `/` ヘルスチェック
- POST `/chat` チャット推論

POST `/chat`
- Request JSON
    - `user_id` string 任意 デフォルト`default_user`
    - `text` string 必須 入力テキスト
    - `role_sheet` object 任意 例: `{ "tone": "元気でフレンドリー" }`
    - `over_hallucination` boolean 任意 生出力寄りのモード
    - `history` array 任意 クライアント側で管理する履歴を送信
        - 要素: `{ role: "user"|"assistant", content: string }`
        - サーバーは直近最大5往復を参照。
    - `compressed_memory` string 任意 圧縮履歴（CHM1 形式）。送ると古い履歴として優先参照されます。
- Response JSON
    - `response` string 生成応答（サニタイズ済み）
    - `debug_info` object デバッグ情報（ルート、思考ログ、プロンプト、利用履歴など）
    - `compressed_memory` string 次回以降に送れる圧縮履歴（CHM1）

例（curl）
```bash
curl -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "alice",
        "text": "今日の天気は？",
        "history": [
            {"role":"user","content":"おはよう"},
            {"role":"assistant","content":"やっほー！今日もがんばろうね！"}
        ]
    }'
```

（注）サーバーはステートレスです。履歴の保持や削除APIはありません。履歴はクライアント側で管理してください。

## 6. クライアント側で履歴を管理する方法

`client.py` はセッションの `messages`（role/content）をそのまま `/chat` の `history` に同梱して送信します。サーバーはこの履歴を優先的にコンテキストとして使用します。これにより、複数クライアントや別セッション間の混線を避けつつ、UI側で履歴制御が可能です。

ポイント
- 履歴は `[{role, content}, ...]` の配列で送り、不要な内部フィールドは含めない
- 直近5往復はそのまま参照、それ以前は `compressed_memory`（CHM1）を優先利用
- サーバーは履歴を保持しません（完全ステートレス）。

環境変数
- `SPUNKY_HISTORY_COMPRESSED_MAXTOKENS`（任意）: 直近5件以外を圧縮して保持する際の概算トークン上限（デフォルト 500）。

圧縮履歴（CHM1: Chat History Minimal v1）
- 形式（例）
```
#CHM1 v1
user:alice
RECENT:
U:前回の質問…
A:前回の回答…
---
SUMMARY: 過去の長い会話の要点を1行に圧縮
KEY: 検索,設定,ログ,API,日本語
LAST U:今回のユーザー入力 | A:今回の返答
```
- クライアントは `compressed_memory` として次回リクエストにそのまま送付可能。
- サーバー側は、`history` と `compressed_memory` が重複する場合、CHM1を優先します。

## 7. NLUモデル（ルーター用 RoBERTa）の追加学習

使用モデル: `nlp-waseda/roberta-base-japanese` ベースの意図/感情の分類モデル。保存先は以下。
- 意図: `spunky/memory/nlu_model_intent`
- 感情: `spunky/memory/nlu_model_emotion`

前提
- Python 3.10 以上
- `pip install -r requirements.txt`

学習データ
- CSV/TSVなどで `text,label` の形式を用意（日本語）
- クラス名はラベルにそのまま使用（例: intent: `question|greeting|statement|farewell` 等、emotion: `happy|sad|angry|neutral` 等）

学習手順（例: Transformers Trainer を想定）
1. データを `data/intent_train.csv`, `data/intent_eval.csv` のように分割
2. 簡易トレーニングスクリプトを用意（例）
     - ベース: `nlp-waseda/roberta-base-japanese`
     - max_len=256, batch=16, epoch=3〜5, lr=2e-5 目安
3. 学習後、以下の構成で保存
     - `spunky/memory/nlu_model_intent/` に `config.json`, `pytorch_model.bin`, `tokenizer.json` 等
     - 同様に感情モデルを `spunky/memory/nlu_model_emotion/` へ
4. サーバー再起動で自動ロードされます

Tips
- 環境変数 `SPUNKY_NLU_DEVICE=cuda` でGPUを使用
- クラス名は英小文字を推奨（ルーター側の判定と整合）
- 迷う場合はまず intent に `question` と `greeting` の識別精度を上げるところから始める

（将来）専用スクリプトの整備
- 需要があれば `spunky/scripts/train_nlu.py` を追加し、上記を自動化します

## 知識検索（RAG: Retrieval-Augmented Generation）の仕組み

SpunkyのRAGは、単なるベクトル検索ではなく「エージェント式RAG」として設計されています。主な特徴は以下の通りです。

- **1. LLMによる多様な検索クエリ生成**  
  ユーザーの質問から、LLM（Qwen-LoRA）が4つ前後の多様な検索クエリ（例：「ガールズ&パンツァー オープニング」「Girls und Panzer OP」「ガルパン 主題歌」など）を自動生成します。  
  これにより、単一クエリでは拾えない関連知識も幅広くカバーします。

- **2. FAISS＋SentenceTransformerによる高速ベクトル検索**  
  各クエリごとにFAISS＋埋め込みモデルで最大8件ずつ検索し、重複を除去してマージします。  
  検索結果の全チャンク（最大32件程度）をログ出力し、デバッグや精度検証が容易です。

- **3. ドメイン特化の再ランキング**  
  検索結果は「タイトル・本文のキーワード一致」「n-gram類似度」「ベクトル距離」「テキスト長」など複数指標でスコアリング。  
  さらに、ガルパン（Girls und Panzer）など特定ドメインのコンテンツには最大100点のボーナスを与え、関連性の高い知識を最優先で選択します。

- **4. 全検索結果の詳細ログ**  
  どのクエリで何件ヒットし、どのようなスコアで再ランキングされたかを全てログ出力。  
  これにより、なぜその知識が選ばれたかを後から検証できます。

- **5. 柔軟なバックエンド**  
  LM StudioやOllama、ローカルGGUFなど複数のLLMバックエンドに対応。  
  `SPUNKY_BACKEND=lmstudio` かつ `SPUNKY_STRICT_BACKEND=1` でLM Studioのみを強制使用できます。

**例：ガルパンのオープニング曲を質問した場合**  
→ LLMが「ガールズ&パンツァー オープニング」「Girls und Panzer OP」などのクエリを自動生成  
→ 正確な知識を自然な日本語で返答

---

この仕組みにより、従来のRAGよりも「自然で正確な知識検索」と「なぜその知識が選ばれたかの透明性」を両立しています。
