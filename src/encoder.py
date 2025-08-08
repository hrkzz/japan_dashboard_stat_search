import numpy as np
import faiss
from litellm import embedding
import os
from typing import List, Union
from config import config

class EmbeddingConfig:
    """エンベディングの設定と生成を管理するクラス"""
    
    def __init__(self):
        self.embedding_model = None
    
    def _get_embedding_model(self) -> str:
        """使用するエンベディングモデルを決定"""
        # APIキーを設定
        api_key = self._get_api_key()
        
        # Ollamaが利用可能な場合は優先的に使用
        if api_key.get('ollama'):
            return "ollama/nomic-embed-text"
        elif api_key.get('openai'):
            os.environ["OPENAI_API_KEY"] = api_key['openai']
            return "text-embedding-3-small"
        elif api_key.get('gemini'):
            os.environ["GEMINI_API_KEY"] = api_key['gemini']
            return "text-embedding-004"
        else:
            return "text-embedding-3-small"  # デフォルト
    
    def _get_api_key(self):
        """APIキーを取得（Config 経由）"""
        return {
            'openai': config.get_openai_key(),
            'gemini': config.get_gemini_key(),
            'ollama': config.get_ollama_base_url(),
        }
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """テキストのエンベディングを生成"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # モデルが未設定の場合は設定
            if not self.embedding_model:
                self.embedding_model = self._get_embedding_model()
            
            # embedding関数の引数を準備
            embedding_args = {
                "model": self.embedding_model,
                "input": texts
            }
            
            # Ollamaモデルの場合はapi_baseを追加
            if self.embedding_model.startswith('ollama/'):
                api_keys = self._get_api_key()
                if api_keys.get('ollama'):
                    embedding_args["api_base"] = api_keys['ollama']
            
            response = embedding(**embedding_args)
            
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
                return np.array([])
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            return embeddings_array
            
        except Exception as e:
            # 例外は空配列で返す（UI 非依存）
            return np.array([])
    
    def get_single_embedding(self, text: str) -> np.ndarray:
        """単一テキストのエンベディングを生成"""
        embeddings = self.get_embeddings(text)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

# グローバルインスタンス
embedding_config = EmbeddingConfig()