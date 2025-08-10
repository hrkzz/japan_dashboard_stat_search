import os
from litellm import completion
from typing import Optional, Dict, Any
import ollama
from config import config

class LLMConfig:
    """LLMの設定と初期化を管理するクラス"""
    
    def __init__(self):
        self.api_keys = {}
        self.current_model = None
        self.ollama_models = []
        self.setup_api_keys()
    
    def setup_api_keys(self):
        """利用可能なAPIキーをすべて設定"""
        self.api_keys = self._get_api_keys()
        
        # Ollamaモデルの取得を試行
        self._fetch_ollama_models()
        
        # デバッグ: APIキーの取得状況をログ出力
        print(f"🔧 取得したAPIキー: {list(self.api_keys.keys())}")
        for key_type, key_value in self.api_keys.items():
            print(f"  {key_type}: {'設定済み' if key_value else '未設定'} ({len(key_value) if key_value else 0}文字)")
        
        # Ollamaモデルの取得状況をログ出力
        if self.ollama_models:
            model_names = [name for name in (self._extract_ollama_model_name(m) for m in self.ollama_models) if name]
            print(f"🦙 Ollama利用可能モデル: {model_names}")
        
        # デフォルトモデル: ユーザー環境では Gemini を優先（高速）
        if self.api_keys.get('gemini'):
            self.current_model = "gemini-2.0-flash-exp"
            print("🚀 デフォルトモデル: Gemini 2.0 Flash")
        elif self.api_keys.get('openai'):
            self.current_model = "gpt-4o-mini"
            print("🚀 デフォルトモデル: OpenAI GPT-4o-mini")
        elif self.ollama_models:
            first_name = self._extract_ollama_model_name(self.ollama_models[0])
            if first_name:
                self.current_model = f"ollama/{first_name}"
                print(f"🚀 デフォルトモデル: Ollama {first_name}")
            else:
                self.current_model = None
        else:
            self.current_model = None
            print("❌ 利用可能なAPIキーがありません")
    
    def _fetch_ollama_models(self):
        """Ollamaから利用可能なモデルリストを取得"""
        if not self.api_keys.get('ollama'):
            return
        
        try:
            # Ollama クライアントを初期化
            client = ollama.Client(host=self.api_keys['ollama'])
            models_response = client.list()
            
            # モデルリストを保存
            if 'models' in models_response:
                self.ollama_models = models_response['models']
                print(f"✅ Ollama接続成功: {len(self.ollama_models)}個のモデルが利用可能")
            else:
                print("⚠️ Ollamaからモデルリストを取得できませんでした")
        except Exception as e:
            print(f"⚠️ Ollama接続エラー: {str(e)}")
            self.ollama_models = []
    
    def get_available_models(self) -> Dict[str, str]:
        """利用可能なモデルのリストを返す"""
        available_models = {}
        
        if self.api_keys.get('openai'):
            available_models["OpenAI GPT-4o-mini"] = "gpt-4o-mini"
        
        if self.api_keys.get('gemini'):
            available_models["Google Gemini 2.0 Flash"] = "gemini-2.0-flash-exp"
        
        # Ollamaモデルを追加
        for model in self.ollama_models:
            model_name = self._extract_ollama_model_name(model)
            if not model_name:
                continue
            display_name = f"Ollama: {model_name}"
            available_models[display_name] = f"ollama/{model_name}"
        
        return available_models
    
    def set_model(self, model_name: str):
        """使用するモデルを設定"""
        self.current_model = model_name
        
        # 環境変数を設定
        if model_name.startswith("gpt"):
            os.environ["OPENAI_API_KEY"] = self.api_keys['openai']
        elif model_name.startswith("gemini"):
            os.environ["GEMINI_API_KEY"] = self.api_keys['gemini']
        # Ollamaの場合は環境変数設定不要（api_baseで直接指定）
    
    def _get_api_keys(self):
        """APIキーを取得（環境変数 > secrets.toml を `Config` から）"""
        return {
            'openai': config.get_openai_key(),
            'gemini': config.get_gemini_key(),
            'ollama': config.get_ollama_base_url(),
        }
    
    def generate_response(self, messages: list, temperature: float = 0.3) -> str:
        """LLMからレスポンスを生成"""
        if not self.current_model:
            return "エラー: モデルが選択されていません"
        
        try:
            # モデル名をlitellm形式に変換
            litellm_model = self.current_model
            if self.current_model.startswith("gemini"):
                litellm_model = f"gemini/{self.current_model}"
            
            # completion関数の引数を準備
            completion_args = {
                "model": litellm_model,
                "messages": messages,
                "temperature": temperature
            }
            
            # Ollamaモデルの場合はapi_baseを追加
            if self.current_model.startswith('ollama/'):
                completion_args["api_base"] = self.api_keys['ollama']
            
            response = completion(**completion_args)
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM応答生成中にエラーが発生しました: {str(e)}"

    def generate_response_stream(self, messages: list, temperature: float = 0.3):
        """LLMからのストリーミング応答を生成するジェネレータを返す。"""
        if not self.current_model:
            yield "エラー: モデルが選択されていません"
            return

        try:
            litellm_model = self.current_model
            if self.current_model.startswith("gemini"):
                litellm_model = f"gemini/{self.current_model}"

            completion_args = {
                "model": litellm_model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if self.current_model.startswith("ollama/"):
                completion_args["api_base"] = self.api_keys['ollama']

            for chunk in completion(**completion_args):
                # LiteLLMのチャンク互換処理
                piece = None
                try:
                    piece = chunk.choices[0].delta.content  # OpenAI形式
                except Exception:
                    try:
                        piece = chunk.choices[0].message.get("content")
                    except Exception:
                        piece = None
                if piece:
                    yield piece
        except Exception as e:
            yield f"[stream error]: {str(e)}"

    def _extract_ollama_model_name(self, model: Any) -> Optional[str]:
        """Ollamaの`client.list()`で返るモデル要素からモデル名を取り出す。
        - dict型の場合: 'name' 優先、なければ 'model' を参照
        - オブジェクトの場合: `.name` 優先、なければ `.model` を参照
        取得できなければ None を返す。
        """
        try:
            # dict 互換
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
                if isinstance(name, str) and name.strip():
                    return name.strip()
                return None
            # 属性アクセス
            name_attr = getattr(model, "name", None) or getattr(model, "model", None)
            if isinstance(name_attr, str) and name_attr.strip():
                return name_attr.strip()
            # __getitem__ に対応した型（ollama._types）の可能性
            try:
                name_key = None
                for key in ("name", "model"):
                    try:
                        value = model[key]  # type: ignore[index]
                        if isinstance(value, str) and value.strip():
                            name_key = value.strip()
                            break
                    except Exception:
                        continue
                return name_key
            except Exception:
                return None
        except Exception:
            return None

# グローバルインスタンス
llm_config = LLMConfig()

    