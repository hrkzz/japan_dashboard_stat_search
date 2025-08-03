"""
BigQuery ロギングモジュール
ユーザーの行動ログをGoogle BigQueryに記録する機能を提供
"""

import streamlit as st
import json
from datetime import datetime, timezone
from loguru import logger
from typing import Optional

# BigQueryライブラリの防御的インポート
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None
    service_account = None


class BigQueryLogger:
    """BigQueryロガークラス"""
    
    def __init__(self):
        self.client = None
        self.table_ref_str = None
        self._initialize_client()
    
    def _initialize_client(self):
        """BigQueryクライアントを初期化"""
        if not BIGQUERY_AVAILABLE:
            logger.info("ℹ️ google-cloud-bigqueryが未インストールのため、ログ機能を無効にします。")
            return

        try:
            # st.secretsのキーをデバッグ表示
            logger.debug(f"st.secretsのキー: {list(st.secrets.keys())}")

            # secrets.tomlに必要なセクションがあるか確認
            if "gcp_service_account" in st.secrets and "bigquery" in st.secrets:
                
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                
                bq_config = st.secrets.bigquery
                project_id = bq_config.get("project")
                dataset_id = bq_config.get("dataset")
                table_id = bq_config.get("table")
                
                if not all([project_id, dataset_id, table_id]):
                    logger.warning("⚠️ [bigquery]セクションのproject, dataset, tableキーが不足しています。")
                    return

                self.client = bigquery.Client(
                    credentials=credentials,
                    project=project_id
                )
                
                self.table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
                logger.info(f"✅ BigQueryクライアントが正常に初期化されました。ログはテーブル '{self.table_ref_str}' に送信されます。")
            else:
                logger.warning("⚠️ secrets.tomlに[gcp_service_account]または[bigquery]セクションが見つかりません。ログは無効です。")
                
        except Exception as e:
            logger.error(f"❌ BigQueryクライアント初期化中に予期せぬエラーが発生しました: {str(e)}")
            self.client = None
    
    def log_event(self, **kwargs) -> bool:
        """イベントをBigQueryに記録"""
        if not self.client:
            logger.warning("⚠️ BigQueryクライアント未初期化のため、ログ記録をスキップします。")
            return False
            
        try:
            row_data = kwargs
            row_data["log_timestamp"] = datetime.now(timezone.utc)
            
            # final_indicatorsがリスト/辞書の場合、JSON文字列に変換
            if "final_indicators" in row_data and isinstance(row_data["final_indicators"], (list, dict)):
                row_data["final_indicators"] = json.dumps(row_data["final_indicators"], ensure_ascii=False)
            
            errors = self.client.insert_rows_json(self.table_ref_str, [row_data])
            
            if not errors:
                logger.info(f"✅ ログ記録成功: {kwargs.get('event_type')} - session: {str(kwargs.get('session_id'))[:8]}...")
                return True
            else:
                logger.error(f"❌ BigQueryへのデータ挿入に失敗しました: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ログ記録中に予期せぬエラーが発生しました: {str(e)}")
            return False

# シングルトンインスタンスを管理
@st.cache_resource
def get_bq_logger():
    return BigQueryLogger()

# アプリから呼び出すための便利な関数
def log_event(**kwargs):
    logger_instance = get_bq_logger()
    return logger_instance.log_event(**kwargs)