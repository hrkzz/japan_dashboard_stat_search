# 統計指標検索ダッシュボード（PowerBI用）

このアプリケーションは、政府統計データの指標をセマンティック検索とキーワード検索を組み合わせたハイブリッド検索で効率的に検索し、Power BIダッシュボードで使用する項目を特定することができます。

## 🏗️ アーキテクチャ

### 新しいモジュラー構造

アプリケーションは以下のような構造に分割されています：

```
├── src/
│   ├── llm_config.py        # LLM設定
│   ├── encoder.py           # エンベディング設定とRAG用LLM設定
│   ├── retriever.py         # ハイブリッド検索とリランキング
│   ├── app.py              # Streamlit UI（メイン）
│   └── build_vector_db.py  # ベクトルDB構築スクリプト
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

#### 4. ベクトルデータベースの構築
```bash
cd src
python build_vector_db.py
```

#### 5. アプリケーションの実行
```bash
cd src
streamlit run app.py
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