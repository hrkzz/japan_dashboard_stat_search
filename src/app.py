import streamlit as st
import json
import re
from retriever import retriever
from llm_config import llm_config
from loguru import logger
import time

# 対話の段階を定義
STAGE_INITIAL = "initial"
STAGE_PERSPECTIVE_SELECTION = "perspective_selection"
STAGE_INDICATOR_SELECTION = "indicator_selection"
STAGE_FINAL = "final"

def initialize_session_state():
    """セッション状態を初期化"""
    if 'stage' not in st.session_state:
        st.session_state.stage = STAGE_INITIAL
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_options' not in st.session_state:
        st.session_state.current_options = []
    if 'selected_perspective' not in st.session_state:
        st.session_state.selected_perspective = None
    if 'original_query' not in st.session_state:
        st.session_state.original_query = ""
    if 'available_indicators' not in st.session_state:
        st.session_state.available_indicators = ""
    if 'selected_indicator' not in st.session_state:
        st.session_state.selected_indicator = None

def add_message_to_history(role, content):
    """チャット履歴にメッセージを追加"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def get_available_indicators_for_query(query):
    """クエリに関連する指標を検索し、LLMに渡すためのプロンプト用テキストを生成する"""
    try:
        if retriever.df is None:
            retriever.load_vector_database()
        
        logger.info(f"🔍 指標例取得開始: '{query}'")
        search_results = retriever.hybrid_search(query, top_k=40)
        
        bunya_groups = {}
        for result in search_results:
            bunya = result['bunya_name'] 
            if bunya not in bunya_groups:
                bunya_groups[bunya] = []
            bunya_groups[bunya].append(result['koumoku_name_full'])
        
        for bunya in bunya_groups.keys():
            bunya_indicators = retriever.df[retriever.df['bunya_name'] == bunya]['koumoku_name_full'].tolist()
            existing = set(bunya_groups[bunya])
            additional = [ind for ind in bunya_indicators if ind not in existing][:10]
            bunya_groups[bunya].extend(additional)
        
        indicator_examples = []
        for bunya, indicators in bunya_groups.items():
            indicator_examples.append(f"【{bunya}】({len(indicators)}件利用可能): {', '.join(indicators[:15])}")
        
        return "\n".join(indicator_examples)
    except Exception as e:
        return f"指標リスト取得エラー: {str(e)}"

def generate_analysis_perspectives(query):
    """分析観点を生成するプロンプト1"""
    logger.info(f"🤖 分析観点生成開始: '{query}'")
    available_indicators = get_available_indicators_for_query(query)
    st.session_state.available_indicators = available_indicators
    
    system_prompt = f"""あなたは統計分析の専門家です。ユーザーの質問を分析し、4-5個の分析観点（中項目）を提示してください。

**重要**: 以下の実在する統計指標からのみ観点を設定してください。
利用可能な統計指標：
{available_indicators}

出力は必ずJSON形式で、以下の構造に従ってください：
{{
  "perspectives": [
    {{
      "title": "分析観点のタイトル",
      "description": "この観点で分析する理由の説明"
    }}
  ]
}}

**必須要件**：
- 4-5個の分析観点を提示してください
- 各観点は上記の実在する統計指標に基づいている必要があります
- JSON形式以外は出力しないでください"""

    user_prompt = f"以下の質問について、統計分析の観点から4-5個の分析観点を提示してください：\n\n{query}"
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.2)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            logger.error(f"❌ 有効なJSONが生成されませんでした: {response[:500]}...")
            return None
    except Exception as e:
        st.error(f"分析観点生成でエラーが発生しました: {str(e)}")
        return None

def generate_indicators_for_perspective(perspective_title, available_indicators):
    """選択された観点に基づいて具体的な指標を生成するプロンプト2"""
    logger.info(f"🤖 指標絞り込み開始: '{perspective_title}'")
    
    system_prompt = f"""あなたは統計分析の専門家です。指定された分析観点に関連する具体的な統計指標を提示してください。

**重要**: 以下の実在する統計指標からのみ選択してください。
利用可能な統計指標：
{available_indicators}

出力は必ずJSON形式で、以下の構造に従ってください：
{{
  "indicators": [
    {{
      "indicator_name": "実在する統計指標名（上記リストから選択）",
      "reason": "なぜこの指標を推奨するのかの理由"
    }}
  ]
}}

**必須要件**：
- indicator_nameは上記リストの実在する指標名と完全に一致させてください
- 4-8個程度の指標を提示してください
- JSON形式以外は出力しないでください"""

    user_prompt = f"分析観点「{perspective_title}」に関連する具体的な統計指標を提示してください。"
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.2)
        
        logger.info(f"🔍 デバッグ - LLM応答の最初の500文字: {response[:500]}")
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group())
            logger.info(f"🔍 デバッグ - パースされたJSON: {parsed_json}")
            return parsed_json
        else:
            logger.error(f"❌ 有効なJSONが生成されませんでした: {response[:500]}...")
            # フォールバック：手動で指標リストを生成
            return create_fallback_indicators(available_indicators, perspective_title)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析エラー: {str(e)}")
        return create_fallback_indicators(available_indicators, perspective_title)
    except Exception as e:
        logger.error(f"❌ 指標生成エラー: {str(e)}")
        return create_fallback_indicators(available_indicators, perspective_title)

def create_fallback_indicators(available_indicators, perspective_title):
    """LLMでの指標生成に失敗した場合のフォールバック"""
    logger.info(f"🔄 フォールバック指標生成: '{perspective_title}'")
    
    # 利用可能な指標から最初の5つを抽出
    lines = available_indicators.split('\n')
    indicators = []
    count = 0
    
    for line in lines:
        if count >= 5:
            break
        # 【分野名】(件数)という行をスキップ
        if line.startswith('【') and line.endswith('）'):
            continue
        # 指標名を抽出
        parts = line.split(': ')
        if len(parts) > 1:
            indicator_names = parts[1].split(', ')
            for name in indicator_names:
                if count >= 5:
                    break
                indicators.append({
                    "indicator_name": name.strip(),
                    "reason": f"「{perspective_title}」に関連する指標として推奨"
                })
                count += 1
    
    return {"indicators": indicators}

# interpret_user_choice関数は不要になったため削除

def get_indicator_details(indicator_name):
    """指標名から、DataFrameに格納された詳細情報を取得する"""
    try:
        if retriever.df is None: 
            return None
        
        row = None
        df = retriever.df
        
        exact_matches = df[df['koumoku_name_full'].str.strip() == indicator_name.strip()]
        if not exact_matches.empty: 
            row = exact_matches.iloc[0]
        else:
            partial_matches = df[df['koumoku_name_full'].str.contains(indicator_name.strip(), na=False, case=False)]
            if not partial_matches.empty: 
                row = partial_matches.iloc[0]

        if row is None: 
            return None
        
        return {
            'koumoku_name_full': row.get('koumoku_name_full', ''),
            'bunya_name': row.get('bunya_name', ''),
            'chuubunrui_name': row.get('chuubunrui_name', ''),
            'shoubunrui_name': row.get('shoubunrui_name', ''),
            'koumoku_code': row.get('koumoku_code', '')
        }
    except Exception as e:
        st.error(f"指標詳細取得エラー: {str(e)}")
        return None

def display_indicator_card(indicator_data, recommendation_reason, category_key, indicator_index):
    """単一の指標情報をカード形式で表示する"""
    if not indicator_data:
        st.error("指標データが無効です")
        return

    with st.container():
        col_icon, col_content, col_actions = st.columns([0.3, 5.5, 0.2])

        with col_icon:
            st.markdown("📊")

        with col_content:
            indicator_code = indicator_data.get("koumoku_code", "")
            st.markdown(
                f'<div class="indicator-title">{indicator_data["koumoku_name_full"]} '
                f'<span class="indicator-code">{indicator_code.lstrip("#")}</span></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="indicator-reason">💡 {recommendation_reason}</div>',
                unsafe_allow_html=True
            )
            path = f'{indicator_data["bunya_name"]} > {indicator_data["chuubunrui_name"]} > {indicator_data["shoubunrui_name"]}'
            st.markdown(
                f'<div class="indicator-path">{path}</div>',
                unsafe_allow_html=True
            )

        with col_actions:
            base_url = "https://app.powerbi.com/groups/f57d1ec6-4658-47f7-9a93-08811e43127f/reports/1accacdd-98d0-4d03-9b25-48f4c9673ff4/02fa5822008e814cf7f2?experience=power-bi"
            
            indicator_code = indicator_data.get("koumoku_code", "")
            cleaned_indicator_code = indicator_code.lstrip('#')
            power_bi_url = f"{base_url}&filter=social_demographic_pref_basic_bi/cat3_code eq '{cleaned_indicator_code}'"

            st.markdown(
                f'<div style="text-align: center; padding-top: 5px;"><a href="{power_bi_url}" target="_blank" rel="noopener noreferrer" title="Power BIを新しいタブで開く">🔗</a></div>',
                unsafe_allow_html=True
            )
        st.markdown('<hr style="margin: 4px 0; border: 0.5px solid #e0e0e0;">', unsafe_allow_html=True)

def handle_initial_stage():
    """初期段階：ユーザーからの最初の質問を受け付け"""
    st.markdown("## 📊 統計指標検索アシスタント")
    st.markdown("どのような統計指標をお探しですか？分析したいテーマを下のチャット欄に入力してください。")
    st.markdown("**例**: 子育て環境を比較したい、地域の教育水準を知りたい、高齢化の現状を把握したい")

def handle_perspective_selection_stage():
    """観点選択段階：分析観点を提示してユーザーに選択してもらう"""
    st.markdown("### 🎯 分析観点の選択")
    st.markdown(f"「{st.session_state.original_query}」について、どのような観点で分析しますか？")
    st.markdown("以下の選択肢からボタンを押してください。")
    
    for i, option in enumerate(st.session_state.current_options):
        # カード形式の美しいデザイン
        st.markdown(f"""
        <div class="selection-card" style="padding: 20px; margin: 16px 0; border-radius: 12px; border: 2px solid #e8f4fd; background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%); box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div class="card-title" style="color: #1f77b4; font-weight: 600; font-size: 1.1em; margin-bottom: 8px;">
                {i+1}. {option['title']}
            </div>
            <div class="card-description" style="color: #666; font-size: 0.9em; line-height: 1.4;">
                {option['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ボタンを中央に配置
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("この観点を選択", key=f"perspective_{i}", type="primary", use_container_width=True):
                st.session_state.selected_perspective = option
                add_message_to_history("user", f"{i+1}番目の{option['title']}について詳しく知りたいです")
                
                # 選択された観点に基づいて指標を生成
                with st.spinner("🤖 指標を生成中..."):
                    indicators_result = generate_indicators_for_perspective(
                        option['title'], 
                        st.session_state.available_indicators
                    )
                    
                    if indicators_result and 'indicators' in indicators_result and indicators_result['indicators']:
                        st.session_state.current_options = indicators_result['indicators']
                        st.session_state.stage = STAGE_INDICATOR_SELECTION
                        add_message_to_history("assistant", 
                            f"「{option['title']}」ですね。さらに具体的な指標をご案内します。")
                    else:
                        logger.error(f"❌ 指標生成結果が無効: {indicators_result}")
                        st.error("指標の生成に失敗しました。もう一度お試しください。")
                st.rerun()

def handle_indicator_selection_stage():
    """指標選択段階：具体的な指標を提示してユーザーに選択してもらう"""
    perspective = st.session_state.selected_perspective
    st.markdown("### 📈 具体的な指標の選択")
    st.markdown(f"「{perspective['title']}」について、さらに具体的な指標をご案内します。")
    st.markdown("興味のある指標のボタンを押してください。")
    
    # 指標データが正しく生成されているかチェック
    if not st.session_state.current_options:
        st.error("指標データが取得できませんでした。")
        if st.button("🔄 指標を再生成", key="regenerate_indicators"):
            regenerate_indicators_for_current_perspective()
        return
        
    # データ構造のデバッグと検証
    if st.session_state.current_options and len(st.session_state.current_options) > 0:
        sample_option = st.session_state.current_options[0]
        logger.info(f"🔍 デバッグ - current_optionsの構造: {sample_option}")
        
        # 観点選択段階のデータが残っている場合の対処
        if 'title' in sample_option and 'indicator_name' not in sample_option:
            st.warning("観点選択から指標選択への移行でエラーが発生しました。指標を再生成します...")
            regenerate_indicators_for_current_perspective()
            st.rerun()
            return
    
    for i, option in enumerate(st.session_state.current_options):
        # データが辞書でない場合のハンドリング
        if not isinstance(option, dict):
            st.error(f"指標データの形式が正しくありません: {option}")
            continue
            
        # indicator_nameキーが存在するかチェック
        indicator_name = option.get('indicator_name') or option.get('name') or str(option)
        reason = option.get('reason', '理由が記載されていません')
        
        # カード形式の美しいデザイン
        st.markdown(f"""
        <div class="selection-card" style="padding: 20px; margin: 16px 0; border-radius: 12px; border: 2px solid #e8f4fd; background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%); box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div class="card-title" style="color: #1f77b4; font-weight: 600; font-size: 1.1em; margin-bottom: 8px;">
                {i+1}. {indicator_name}
            </div>
            <div class="card-description" style="color: #666; font-size: 0.9em; line-height: 1.4;">
                {reason}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ボタンを中央に配置
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("この指標を選択", key=f"indicator_{i}", type="primary", use_container_width=True):
                # 最終的な指標が選択された
                indicator_data = get_indicator_details(indicator_name)
                if indicator_data:
                    st.session_state.stage = STAGE_FINAL
                    st.session_state.selected_indicator = {
                        'data': indicator_data,
                        'reason': reason
                    }
                    add_message_to_history("user", f"{indicator_name}の詳細情報が知りたい")
                    add_message_to_history("assistant", f"「{indicator_name}」の詳細情報を表示します。")
                st.rerun()

def regenerate_indicators_for_current_perspective():
    """現在の観点に基づいて指標を再生成"""
    if st.session_state.selected_perspective:
        logger.info(f"🔄 指標再生成: {st.session_state.selected_perspective['title']}")
        indicators_result = generate_indicators_for_perspective(
            st.session_state.selected_perspective['title'], 
            st.session_state.available_indicators
        )
        
        if indicators_result and 'indicators' in indicators_result and indicators_result['indicators']:
            st.session_state.current_options = indicators_result['indicators']
            logger.info(f"✅ 指標再生成成功: {len(indicators_result['indicators'])}件")
        else:
            logger.error(f"❌ 指標再生成失敗: {indicators_result}")
            st.error("指標の再生成に失敗しました。")

def reset_session_state():
    """セッション状態をリセットして新しい検索を開始"""
    logger.info("🔄 セッション状態をリセット")
    for key in ['stage', 'current_options', 'selected_perspective', 'original_query', 'available_indicators', 'selected_indicator']:
        if key in st.session_state:
            del st.session_state[key]
    
    # 初期状態に戻す
    st.session_state.stage = STAGE_INITIAL
    st.session_state.current_options = []

# check_if_new_query関数は不要になったため削除

def handle_final_stage():
    """最終段階：選択された指標の詳細情報を表示"""
    st.markdown("### 📊 指標詳細情報")
    
    if 'selected_indicator' in st.session_state:
        indicator_data = st.session_state.selected_indicator['data']
        reason = st.session_state.selected_indicator['reason']
        display_indicator_card(indicator_data, reason, "final", 0)
    
    st.markdown("---")
    st.markdown("他にお探しの指標はありますか？")
    
    if st.button("🔄 新しい検索を開始", key="new_search", type="primary"):
        reset_session_state()
        st.rerun()

def process_user_input(user_input):
    """ユーザーの入力を処理し、適切な応答を生成（初期段階のみ）"""
    add_message_to_history("user", user_input)
    
    if st.session_state.stage == STAGE_INITIAL:
        # 初期段階：分析観点を生成
        with st.spinner("🤖 分析観点を生成中..."):
            perspectives_result = generate_analysis_perspectives(user_input)
            
            if perspectives_result and 'perspectives' in perspectives_result:
                st.session_state.current_options = perspectives_result['perspectives']
                st.session_state.original_query = user_input
                st.session_state.stage = STAGE_PERSPECTIVE_SELECTION
                
                add_message_to_history("assistant", 
                    f"承知いたしました。「{user_input}」についてですね。どのような観点で分析しますか？")
            else:
                add_message_to_history("assistant", "申し訳ございません。分析観点の生成に失敗しました。もう一度お試しください。")

# display_chat_history関数は削除（会話履歴は不要）

def main():
    """アプリケーションのメインロジック"""
    st.set_page_config(
        page_title="統計指標検索アシスタント", 
        page_icon="📊",
        layout="wide"
    )
    
    # CSSスタイルの追加
    st.markdown("""
        <style>
        .main > div { padding-top: 2rem; }
        .stTextInput > div > div > input { border-radius: 8px; border: 2px solid #e0e0e0; }
        
        /* 指標詳細カード用スタイル */
        .indicator-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin: 4px 0; background-color: #fafafa; transition: box-shadow 0.2s; }
        .indicator-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .indicator-title { font-size: 1.1em; font-weight: bold; color: #1f77b4; margin: 4px 0; }
        .indicator-code { color: #666; font-size: 0.9em; font-weight: bold; }
        .indicator-path { color: #888; font-size: 0.85em; margin: 2px 0; }
        .indicator-reason { color: #f39c12; font-size: 0.9em; margin: 4px 0; }
        
        /* 選択肢カード用スタイル */
        .stContainer > div[data-testid="column"] {
            padding: 0 8px;
        }
        
        .selection-card {
            border-radius: 12px;
            border: 2px solid #e8f4fd;
            background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%);
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .selection-card:hover {
            border-color: #1f77b4;
            box-shadow: 0 4px 12px rgba(31, 119, 180, 0.15);
            transform: translateY(-2px);
        }
        
        .card-title {
            color: #1f77b4;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .card-description {
            color: #666;
            font-size: 0.9em;
            line-height: 1.4;
            margin-top: 8px;
        }
        
        /* ボタンのスタイル調整 */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1f77b4 0%, #1e88e5 100%);
            border: none;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(31, 119, 180, 0.3);
            transition: all 0.2s ease;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
            box-shadow: 0 4px 8px rgba(31, 119, 180, 0.4);
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # セッション状態の初期化
    initialize_session_state()
    
    # データベースの初期化
    with st.spinner("📚 統計データベースを初期化中..."):
        if not retriever.load_vector_database():
            st.error("❌ データベースの読み込みに失敗しました")
            st.stop()
    
    # LLMモデル選択
    available_models = llm_config.get_available_models()
    if available_models:
        with st.sidebar:
            st.markdown("### 🚀 AI設定")
            model_options = list(available_models.keys())
            current_model_display = next((k for k, v in available_models.items() if v == llm_config.current_model), None)
            selected_model_display = st.selectbox(
                "AIモデル選択", model_options,
                index=model_options.index(current_model_display) if current_model_display else 0
            )
            selected_model = available_models[selected_model_display]
            if selected_model != llm_config.current_model:
                llm_config.set_model(selected_model)
    
    # メインコンテンツ（上から下の流れ）
    if st.session_state.stage == STAGE_INITIAL:
        handle_initial_stage()
        
        # 初期段階のみチャット入力を表示
        user_input = st.chat_input("分析したいテーマを入力してください（例：子育て環境を比較したい）")
        if user_input:
            process_user_input(user_input)
            st.rerun()
            
    elif st.session_state.stage == STAGE_PERSPECTIVE_SELECTION:
        handle_perspective_selection_stage()
    elif st.session_state.stage == STAGE_INDICATOR_SELECTION:
        handle_indicator_selection_stage()
    elif st.session_state.stage == STAGE_FINAL:
        handle_final_stage()
    
    # 会話履歴は表示しない

if __name__ == "__main__":
    main()