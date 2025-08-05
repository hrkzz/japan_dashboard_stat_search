"""
BigQuery ロギングモジュール (完全最終版)
"""

import streamlit as st
import json
from datetime import datetime, timezone
from loguru import logger
import pytz

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

class BigQueryLogger:
    def __init__(self):
        self.client = None
        self.table_ref = None
        self._initialize_client()

    def _initialize_client(self):
        if not BIGQUERY_AVAILABLE: return
        try:
            if "gcp_service_account" in st.secrets and "bigquery" in st.secrets:
                credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
                bq_config = st.secrets.bigquery
                project_id, dataset_id, table_id = bq_config.get("project"), bq_config.get("dataset"), bq_config.get("table")

                if not all([project_id, dataset_id, table_id]):
                    logger.warning("secrets.tomlの[bigquery]セクションにproject, dataset, tableのいずれかがありません。")
                    return

                self.client = bigquery.Client(credentials=credentials, project=project_id)
                self.table_ref = self.client.dataset(dataset_id).table(table_id)
                logger.info("✅ BigQueryクライアントが正常に初期化されました。")
            else:
                logger.warning("secrets.tomlに[gcp_service_account]または[bigquery]の設定が見つかりません。")
        except Exception as e:
            logger.error(f"❌ BigQueryクライアント初期化エラー: {e}")
            self.client = None

    def log_event(self, **kwargs):
        if not self.client: return

        try:
            jst = pytz.timezone('Asia/Tokyo')
            row_to_insert = {"log_timestamp": datetime.now(jst).isoformat()}
            
            for key, value in kwargs.items():
                # あらゆる値を文字列に変換する
                if value is None:
                    row_to_insert[key] = None
                elif isinstance(value, (dict, list)):
                    row_to_insert[key] = json.dumps(value, ensure_ascii=False, default=str)
                else:
                    row_to_insert[key] = str(value)

            errors = self.client.insert_rows_json(self.table_ref, [row_to_insert])
            if not errors:
                logger.info(f"✅ ログ記録成功: {row_to_insert.get('event_type')}")
            else:
                logger.error(f"❌ BigQuery挿入エラー: {errors}")
        except Exception as e:
            logger.error(f"❌ ログ記録中に予期せぬエラー: {e}")

_bq_logger_instance = BigQueryLogger()

def log_event(**kwargs):
    _bq_logger_instance.log_event(**kwargs)