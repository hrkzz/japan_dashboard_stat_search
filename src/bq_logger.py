"""
BigQuery ロギングモジュール
ユーザーの行動ログをGoogle BigQueryに記録する機能を提供
"""

import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import json
from datetime import datetime, timezone
from loguru import logger
from typing import Dict, Optional


class BigQueryLogger:
    """BigQueryロガークラス"""
    
    def __init__(self):
        self.client = None
        self.project_id = None
        self.dataset_id = None
        self.table_id = None
        self._initialize_client()
    
    def _initialize_client(self):
        """BigQueryクライアントを初期化"""
        try:
            # Streamlit secretsから設定を読み込み
            if "gcp_service_account" in st.secrets:
                # サービスアカウント情報を取得
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                
                # BigQuery設定を取得
                self.project_id = st.secrets.get("bigquery_project_id")
                self.dataset_id = st.secrets.get("bigquery_dataset_id", "user_logs")
                self.table_id = st.secrets.get("bigquery_table_id", "app_activity")
                
                # BigQueryクライアントを初期化
                self.client = bigquery.Client(
                    credentials=credentials,
                    project=self.project_id
                )
                
                logger.info("✅ BigQueryクライアントが正常に初期化されました")
            else:
                logger.warning("⚠️ BigQuery設定が見つかりません（secrets.tomlを確認してください）")
                
        except Exception as e:
            logger.error(f"❌ BigQueryクライアント初期化エラー: {str(e)}")
            self.client = None
    
    def log_event(self, 
                  session_id: str,
                  event_type: str,
                  user_query: Optional[str] = None,
                  selected_perspective: Optional[str] = None,
                  selected_group: Optional[str] = None,
                  final_indicators: Optional[list] = None,
                  llm_model: Optional[str] = None) -> bool:
        """
        イベントをBigQueryに記録
        
        Args:
            session_id: セッションID
            event_type: イベントタイプ ('query', 'selection' など)
            user_query: ユーザーのクエリ
            selected_perspective: 選択された分析観点
            selected_group: 選択された指標グループ
            final_indicators: 最終的な指標リスト
            llm_model: 使用されたLLMモデル
            
        Returns:
            bool: ログ記録が成功したかどうか
        """
        try:
            # BigQueryクライアントが初期化されていない場合はスキップ
            if not self.client or not self.project_id:
                logger.warning("⚠️ BigQueryクライアントが初期化されていません。ログ記録をスキップします。")
                return False
            
            # 現在のタイムスタンプを生成
            log_timestamp = datetime.now(timezone.utc)
            
            # final_indicatorsをJSON文字列に変換
            final_indicators_json = None
            if final_indicators:
                try:
                    # 指標名のリストを作成
                    indicator_names = [
                        indicator.get('koumoku_name_full', '') 
                        for indicator in final_indicators
                    ]
                    final_indicators_json = json.dumps(indicator_names, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"⚠️ 指標リストのJSON変換エラー: {str(e)}")
                    final_indicators_json = str(final_indicators)
            
            # ログデータを準備
            row_data = {
                "log_timestamp": log_timestamp,
                "session_id": session_id,
                "event_type": event_type,
                "user_query": user_query,
                "selected_perspective": selected_perspective,
                "selected_group": selected_group,
                "final_indicators": final_indicators_json,
                "llm_model": llm_model
            }
            
            # テーブル参照を作成
            table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
            table = self.client.get_table(table_ref)
            
            # データを挿入
            errors = self.client.insert_rows_json(table, [row_data])
            
            if errors:
                logger.error(f"❌ BigQuery挿入エラー: {errors}")
                return False
            else:
                logger.info(f"✅ ログ記録成功: {event_type} - セッション: {session_id[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"❌ ログ記録エラー: {str(e)}")
            return False


# グローバルインスタンス
_bq_logger = None

def get_logger():
    """BigQueryロガーのシングルトンインスタンスを取得"""
    global _bq_logger
    if _bq_logger is None:
        _bq_logger = BigQueryLogger()
    return _bq_logger

def log_event(session_id: str,
              event_type: str,
              user_query: Optional[str] = None,
              selected_perspective: Optional[str] = None,
              selected_group: Optional[str] = None,
              final_indicators: Optional[list] = None,
              llm_model: Optional[str] = None) -> bool:
    """
    便利な関数：イベントをBigQueryに記録
    
    Args:
        session_id: セッションID
        event_type: イベントタイプ
        user_query: ユーザーのクエリ
        selected_perspective: 選択された分析観点
        selected_group: 選択された指標グループ
        final_indicators: 最終的な指標リスト
        llm_model: 使用されたLLMモデル
        
    Returns:
        bool: ログ記録が成功したかどうか
    """
    logger_instance = get_logger()
    return logger_instance.log_event(
        session_id=session_id,
        event_type=event_type,
        user_query=user_query,
        selected_perspective=selected_perspective,
        selected_group=selected_group,
        final_indicators=final_indicators,
        llm_model=llm_model
    ) 