import numpy as np
import faiss
from litellm import embedding
import streamlit as st
import os
from typing import List, Union

class EmbeddingConfig:
    """エンベディングの設定と生成を管理するクラス"""
    
    def __init__(self):
        self.embedding_model = None
    
    def _get_embedding_model(self) -> str:
        """使用するエンベディングモデルを決定"""
        # APIキーを設定
        api_key = self._get_api_key()
        if api_key.get('openai'):
            os.environ["OPENAI_API_KEY"] = api_key['openai']
            return "text-embedding-3-small"
        elif api_key.get('gemini'):
            os.environ["GEMINI_API_KEY"] = api_key['gemini']
            return "text-embedding-004"
        else:
            return "text-embedding-3-small"  # デフォルト
    
    def _get_api_key(self):
        """APIキーを取得（Streamlit secrets.toml または環境変数）"""
        api_keys = {}
        
        try:
            # Streamlit内での実行の場合
            if "OPENAI_API_KEY" in st.secrets:
                api_keys['openai'] = st.secrets["OPENAI_API_KEY"]
            if "GEMINI_API_KEY" in st.secrets:
                api_keys['gemini'] = st.secrets["GEMINI_API_KEY"]
        except:
            # Streamlit外での実行の場合、直接secrets.tomlファイルを読む
            try:
                import toml
                secrets_path = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml')
                if os.path.exists(secrets_path):
                    with open(secrets_path, 'r') as f:
                        secrets = toml.load(f)
                    if 'OPENAI_API_KEY' in secrets:
                        api_keys['openai'] = secrets['OPENAI_API_KEY']
                    if 'GEMINI_API_KEY' in secrets:
                        api_keys['gemini'] = secrets['GEMINI_API_KEY']
            except ImportError:
                print("tomlライブラリがインストールされていません")
            except Exception as e:
                print(f"secrets.toml読み込みエラー: {e}")
        
        # 環境変数もチェック
        if not api_keys.get('openai') and os.getenv('OPENAI_API_KEY'):
            api_keys['openai'] = os.getenv('OPENAI_API_KEY')
        if not api_keys.get('gemini') and os.getenv('GEMINI_API_KEY'):
            api_keys['gemini'] = os.getenv('GEMINI_API_KEY')
        
        return api_keys
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """テキストのエンベディングを生成"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # モデルが未設定の場合は設定
            if not self.embedding_model:
                self.embedding_model = self._get_embedding_model()
            
            response = embedding(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = []
            # LiteLLMの返り値構造に対応
            if hasattr(response, 'data'):
                # OpenAI形式
                for item in response.data:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
                    elif isinstance(item, dict) and 'embedding' in item:
                        embeddings.append(item['embedding'])
            elif isinstance(response, dict) and 'data' in response:
                # 辞書形式
                for item in response['data']:
                    if isinstance(item, dict) and 'embedding' in item:
                        embeddings.append(item['embedding'])
            else:
                # 直接埋め込みベクトルが返される場合
                if isinstance(response, list):
                    embeddings = response
                else:
                    embeddings = [response]
            
            if not embeddings:
                st.error("エンベディングデータが取得できませんでした")
                return np.array([])
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            return embeddings_array
            
        except Exception as e:
            st.error(f"エンベディング生成エラー: {str(e)}")
            return np.array([])
    
    def get_single_embedding(self, text: str) -> np.ndarray:
        """単一テキストのエンベディングを生成"""
        embeddings = self.get_embeddings(text)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

# グローバルインスタンス
embedding_config = EmbeddingConfig() 