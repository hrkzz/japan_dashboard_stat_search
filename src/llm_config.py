import os
import streamlit as st
from litellm import completion
from typing import Optional, Dict, Any

class LLMConfig:
    """LLMの設定と初期化を管理するクラス"""
    
    def __init__(self):
        self.api_keys = {}
        self.current_model = None
        self.setup_api_keys()
    
    def setup_api_keys(self):
        """利用可能なAPIキーをすべて設定"""
        self.api_keys = self._get_api_keys()
        
        # デバッグ: APIキーの取得状況をログ出力
        print(f"🔧 取得したAPIキー: {list(self.api_keys.keys())}")
        for key_type, key_value in self.api_keys.items():
            print(f"  {key_type}: {'設定済み' if key_value else '未設定'} ({len(key_value) if key_value else 0}文字)")
        
        # デフォルトモデルを設定（OpenAIを優先に変更）
        if self.api_keys.get('openai'):
            self.current_model = "gpt-4o-mini"
            print("🚀 デフォルトモデル: OpenAI GPT-4o-mini")
        elif self.api_keys.get('gemini'):
            self.current_model = "gemini-2.0-flash-exp"
            print("🚀 デフォルトモデル: Gemini 2.0 Flash")
        else:
            self.current_model = None
            print("❌ 利用可能なAPIキーがありません")
    
    def get_available_models(self) -> Dict[str, str]:
        """利用可能なモデルのリストを返す"""
        available_models = {}
        
        if self.api_keys.get('openai'):
            available_models["OpenAI GPT-4o-mini"] = "gpt-4o-mini"
        
        if self.api_keys.get('gemini'):
            available_models["Google Gemini 2.0 Flash"] = "gemini-2.0-flash-exp"
        
        return available_models
    
    def set_model(self, model_name: str):
        """使用するモデルを設定"""
        self.current_model = model_name
        
        # 環境変数を設定
        if model_name.startswith("gpt"):
            os.environ["OPENAI_API_KEY"] = self.api_keys['openai']
        elif model_name.startswith("gemini"):
            os.environ["GEMINI_API_KEY"] = self.api_keys['gemini']
    
    def _get_api_keys(self):
        """APIキーを取得（Streamlit secrets.toml または環境変数）"""
        api_keys = {}
        
        try:
            # Streamlit内での実行の場合
            if "OPENAI_API_KEY" in st.secrets:
                api_keys['openai'] = st.secrets["OPENAI_API_KEY"]
            if "GEMINI_API_KEY" in st.secrets:
                api_keys['gemini'] = st.secrets["GEMINI_API_KEY"]
            if "OLLAMA_BASE_URL" in st.secrets:
                api_keys['ollama'] = st.secrets["OLLAMA_BASE_URL"]
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
                    if 'OLLAMA_BASE_URL' in secrets:
                        api_keys['ollama'] = secrets['OLLAMA_BASE_URL']
            except ImportError:
                print("tomlライブラリがインストールされていません")
            except Exception as e:
                print(f"secrets.toml読み込みエラー: {e}")
        
        # 環境変数もチェック
        if not api_keys.get('openai') and os.getenv('OPENAI_API_KEY'):
            api_keys['openai'] = os.getenv('OPENAI_API_KEY')
        if not api_keys.get('gemini') and os.getenv('GEMINI_API_KEY'):
            api_keys['gemini'] = os.getenv('GEMINI_API_KEY')
        if not api_keys.get('ollama') and os.getenv('OLLAMA_BASE_URL'):
            api_keys['ollama'] = os.getenv('OLLAMA_BASE_URL')
        
        return api_keys
    
    def generate_response(self, messages: list, temperature: float = 0.3) -> str:
        """LLMからレスポンスを生成"""
        if not self.current_model:
            return "エラー: モデルが選択されていません"
        
        try:
            # モデル名をlitellm形式に変換
            litellm_model = self.current_model
            if self.current_model.startswith("gemini"):
                litellm_model = f"gemini/{self.current_model}"
            
            response = completion(
                model=litellm_model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM応答生成中にエラーが発生しました: {str(e)}"

# グローバルインスタンス
llm_config = LLMConfig() 