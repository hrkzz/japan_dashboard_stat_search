.PHONY: help setup install build-db lint run clean activate-help deactivate-help

# デフォルトターゲット
help:
	@echo "利用可能なコマンド:"
	@echo "  setup        - 仮想環境を構築し、依存関係をインストール"
	@echo "  install      - 依存関係のみをインストール"
	@echo "  build-db     - ベクトルデータベースを構築"
	@echo "  lint         - ruffによるコード品質チェック"
	@echo "  run          - Streamlitアプリケーションを実行"
	@echo "  clean        - 仮想環境とキャッシュを削除"
	@echo "  activate-help - 仮想環境をアクティベートする方法を表示"
	@echo "  deactivate-help - 仮想環境をディアクティベートする方法を表示"

# 仮想環境構築と依存関係インストール
setup:
	@echo "仮想環境を構築しています..."
	python -m venv .venv
	@echo "依存関係をインストールしています..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install ruff
	@echo "セットアップが完了しました。"
	@echo "仮想環境をアクティベートするには: make activate-help"

# 依存関係のみをインストール
install:
	@echo "依存関係をインストールしています..."
	pip install -r requirements.txt
	pip install ruff
	@echo "インストールが完了しました。"

# ベクトルデータベース構築
build-db:
	@echo "ベクトルデータベースを構築しています..."
	cd src && python build_vector_db.py
	@echo "ベクトルデータベースの構築が完了しました。"

# Ruffによるlinting
lint:
	@echo "コード品質をチェックしています..."
	ruff check src/
	ruff format --check src/
	@echo "コード品質チェックが完了しました。"

# Ruffによるコード修正
fix:
	@echo "コードを自動修正しています..."
	ruff check --fix src/
	ruff format src/
	@echo "コードの自動修正が完了しました。"

# Streamlitアプリケーション実行
run:
	@echo "Streamlitアプリケーションを起動しています..."
	cd src && streamlit run app.py

# クリーンアップ
clean:
	@echo "クリーンアップを実行しています..."
	rm -rf .venv
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf vector_db
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	@echo "クリーンアップが完了しました。"

# 仮想環境アクティベート方法の表示
activate-help:
	@echo "仮想環境をアクティベートするには、以下のコマンドを実行してください:"
	@echo "  Linux/Mac: source .venv/bin/activate"
	@echo "  Windows:   .venv\\Scripts\\activate"

# 仮想環境ディアクティベート方法の表示
deactivate-help:
	@echo "仮想環境をディアクティベートするには、以下のコマンドを実行してください:"
	@echo "  deactivate" 