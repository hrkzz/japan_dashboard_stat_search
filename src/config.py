"""
アプリ全体の設定値・シークレットの取得を一元化するモジュール。

優先順位:
1) 環境変数
2) Streamlit の st.secrets
3) プロジェクト直下の .streamlit/secrets.toml

UI から独立したスクリプトでも利用できるように設計する。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class Config:
    """設定値の読み出しを提供するクラス。"""

    def __init__(self) -> None:
        self._secrets: Dict[str, Any] = {}
        # st.secrets が読める場合のみ取り込む
        try:
            import streamlit as st  # type: ignore

            # st.secrets は Mapping のように振る舞う
            if hasattr(st, "secrets"):
                try:
                    # 一部の環境では st.secrets を for/keys で辿れないことがあるため個別参照
                    for key in [
                        "OPENAI_API_KEY",
                        "GEMINI_API_KEY",
                        "OLLAMA_BASE_URL",
                        "gcp_service_account",
                        "bigquery",
                        "VECTOR_DB_ZIP_URL",
                    ]:
                        if key in st.secrets:  # type: ignore
                            self._secrets[key] = st.secrets[key]  # type: ignore
                except Exception:
                    pass
        except Exception:
            # Streamlit 非実行時は無視
            pass

        # toml を直接読む（存在する場合のみ）
        try:
            import toml  # type: ignore

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            secrets_path = os.path.join(project_root, ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                secrets = toml.load(secrets_path)
                # st.secrets で既に得られているものは上書きしない
                for key, value in secrets.items():
                    if key not in self._secrets:
                        self._secrets[key] = value
        except Exception:
            # toml が無い/読めない場合は無視
            pass

    # --- LLM 関連キー ---
    def get_openai_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY") or self._get_secret_str("OPENAI_API_KEY")

    def get_gemini_key(self) -> Optional[str]:
        return os.getenv("GEMINI_API_KEY") or self._get_secret_str("GEMINI_API_KEY")

    def get_ollama_base_url(self) -> Optional[str]:
        return os.getenv("OLLAMA_BASE_URL") or self._get_secret_str("OLLAMA_BASE_URL")

    # --- BigQuery/GCP ---
    def get_gcp_service_account_info(self) -> Optional[Dict[str, Any]]:
        """サービスアカウント情報を dict で返す。

        優先順位:
        - 環境変数 GOOGLE_SERVICE_ACCOUNT_JSON / GCP_SERVICE_ACCOUNT_JSON（JSON 文字列）
        - secrets.toml の gcp_service_account セクション
        """
        import json

        for env_key in ("GOOGLE_SERVICE_ACCOUNT_JSON", "GCP_SERVICE_ACCOUNT_JSON"):
            json_str = os.getenv(env_key)
            if json_str:
                try:
                    return json.loads(json_str)
                except Exception:
                    pass

        value = self._secrets.get("gcp_service_account")
        if isinstance(value, dict):
            return value
        return None

    def get_bigquery_config(self) -> Optional[Dict[str, Any]]:
        """BigQuery 用の project/dataset/table 設定を返す。"""
        # 環境変数優先
        env_project = os.getenv("BQ_PROJECT")
        env_dataset = os.getenv("BQ_DATASET")
        env_table = os.getenv("BQ_TABLE")
        if env_project and env_dataset and env_table:
            return {"project": env_project, "dataset": env_dataset, "table": env_table}

        # secrets.toml
        bq = self._secrets.get("bigquery")
        if isinstance(bq, dict):
            return {
                "project": bq.get("project"),
                "dataset": bq.get("dataset"),
                "table": bq.get("table"),
            }
        return None

    # --- Vector DB 配布 ZIP URL ---
    def get_vector_db_zip_url(self) -> str:
        """Vector DB の配布 ZIP URL を返す。環境変数で上書き可能。"""
        return (
            os.getenv("VECTOR_DB_ZIP_URL")
            or self._get_secret_str("VECTOR_DB_ZIP_URL")
            or "https://github.com/hrkzz/japan_dashboard_stat_search/releases/download/v1.0.0/vector_db.zip"
        )

    # --- 汎用 ---
    def _get_secret_str(self, key: str) -> Optional[str]:
        value = self._secrets.get(key)
        if isinstance(value, str):
            return value
        return None


# デフォルトのグローバルインスタンス
config = Config()

