# 統計指標検索ダッシュボード（PowerBI用）

このアプリケーションは、政府統計データの指標をセマンティック検索とキーワード検索を組み合わせたハイブリッド検索で効率的に検索し、Power BIダッシュボードで使用する項目を特定することができます。

## 🏗️ アーキテクチャ

### 新しいモジュラー構造

アプリケーションは以下のような構造に分割されています：

```
├── src/
│   ├── app.py               # (大幅にスリム化) StreamlitのUIエントリ
│   ├── state_manager.py     # (新規) セッション状態管理クラス
│   ├── ui_components.py     # (新規) UI描画関数群
│   ├── services.py          # (新規) ビジネスロジック層
│   ├── config.py            # (新規) 設定と機密情報の一元管理
│   ├── bq_logger.py         # (修正) config.pyを利用
│   ├── encoder.py           # (修正) config.pyを利用
│   ├── llm_config.py        # (修正) config.pyを利用
│   ├── retriever.py         # (修正) config.pyを利用
│   ├── build_vector_db.py   # (修正) UIから独立
│   └── assets/
│       └── style.css        # (新規) CSSファイル
├── requirements.txt        # 依存関係
├── data/                   # 統計データ
└── vector_db/             # 構築済みベクトルDB
```

### 機能分離の利点

1. **LLM設定の柔軟性** (`src/llm_config.py`)
   - OpenAI GPT
   - Google Gemini
   - ローカルLLM (Ollama)
   - LiteLLMによる統一インターフェース

2. **エンベディング対応** (`src/encoder.py`)
   - 複数のエンベディングプロバイダー対応
   - エンベディング生成の最適化

3. **高度な検索機能** (`src/retriever.py`)
   - **ハイブリッド検索**: ベクトル検索 + BM25 + TF-IDF
   - **リランキング**: クエリとの類似度ベースの再順位付け
   - 検索重み調整可能

4. **シンプルなUI** (`src/app.py`)
   - Streamlitによる直感的なインターフェース
   - koumoku_name中心の簡潔な出力
   - 5301個の指標から最適な選択を支援

## 📋 セットアップ

### Makefileを使用した簡単セットアップ（推奨）

```bash
# 1. 全体のセットアップ（仮想環境構築 + 依存関係インストール）
make setup

# 2. 仮想環境をアクティベート
make activate-help  # コマンドを確認
source .venv/bin/activate  # Linux/Mac

# 3. APIキー設定（下記参照）

# 4. ベクトルデータベース構築
make build-db

# 5. アプリケーション実行
make run
```

### 手動セットアップ

#### 1. 仮想環境の準備
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

#### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

#### 3. APIキーの設定

`.streamlit/secrets.toml` ファイルを作成し、使用するLLMプロバイダーのAPIキーを設定：

```toml
# OpenAI使用の場合
OPENAI_API_KEY = "your_openai_api_key"

# または Gemini使用の場合  
GEMINI_API_KEY = "your_gemini_api_key"

# または ローカルLLM (Ollama)使用の場合
OLLAMA_BASE_URL = "http://localhost:11434"
```

##### Ollamaを使用する場合の詳細セットアップ

Ollamaを使用する場合は、以下の手順に従ってセットアップを行ってください：

**1. Ollamaのインストール**

公式サイト（https://ollama.ai/）からOllamaをダウンロードしてインストールしてください。

**2. 使用するモデルのダウンロード**

Ollamaサーバーを起動後、使用したいモデルを事前にダウンロードしておく必要があります：

```bash
# 推奨モデルのダウンロード例
ollama pull llama3        # Meta Llama 3 (チャット用)
ollama pull gemma         # Google Gemma (チャット用)
ollama pull nomic-embed-text  # 埋め込みベクトル生成用

# その他の利用可能なモデル
ollama pull llama3.1      # Meta Llama 3.1
ollama pull codellama     # コード生成特化
ollama pull mistral       # Mistral AI
```

**3. Ollamaサーバーの起動確認**

Ollamaサーバーが正常に起動していることを確認：

```bash
ollama list  # インストール済みモデルの一覧表示
curl http://localhost:11434/api/tags  # API経由でモデル一覧を確認
```

**4. secrets.tomlの設定**

`.streamlit/secrets.toml`にOllama設定を追加：

```toml
OLLAMA_BASE_URL = "http://localhost:11434"
```

**注意事項:**
- Ollamaは初回起動時にモデルをダウンロードするため、インターネット接続が必要です
- モデルサイズが大きいため（数GB〜数十GB）、十分なディスク容量を確保してください
- ローカル実行のため、インターネット接続なしでも利用可能です（モデルダウンロード後）

#### 4. ベクトルデータベースの構築
```bash
python src/build_vector_db.py
```

#### 5. アプリケーションの実行
```bash
streamlit run src/app.py
```

### 🛠️ 開発用コマンド

```bash
# コード品質チェック
make lint

# コード自動修正
make fix

# プロジェクトクリーンアップ
make clean

# 利用可能なコマンド一覧
make help
```

## 🔍 検索機能

### ハイブリッド検索
- **ベクトル検索**: セマンティック類似度による検索
- **BM25検索**: 統計的キーワードマッチング
- **TF-IDF検索**: 語彙の重要度による検索

### リランキング
検索結果をクエリとの関連性で再順位付けし、より精度の高い結果を提供します。

### 検索重み調整
UIでベクトル検索とキーワード検索のバランスを調整できます。

## 🔧 カスタマイズ

### 新しいLLMプロバイダーの追加
`src/llm_config.py`の`setup_api_key()`メソッドに新しいプロバイダーの設定を追加してください。

### 検索アルゴリズムの調整
`src/retriever.py`の各検索メソッドやリランキングロジックを調整できます。

### エンベディングモデルの変更
`src/encoder.py`で使用するエンベディングモデルを変更できます。

## 📊 データソース

- **政府統計の総合窓口（e-Stat）**: https://www.e-stat.go.jp/
- **統計ダッシュボード**: https://dashboard.e-stat.go.jp/

## 🚀 本番環境での運用

Streamlit Cloudや他のクラウドプラットフォームでの実行時は、適切な環境変数やシークレット管理を行ってください。

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Power BIでの活用**: 検索結果の「PowerBI項目名」をそのままPower BIのDAX式やフィルターで使用できます。 