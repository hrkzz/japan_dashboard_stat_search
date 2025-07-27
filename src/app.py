import streamlit as st
import pyperclip
import json
import re
from retriever import retriever
from llm_config import llm_config
from loguru import logger
import time

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
        detailed_search_results = []
        total_indicators = 0
        for bunya, indicators in bunya_groups.items():
            indicator_examples.append(f"【{bunya}】({len(indicators)}件利用可能): {', '.join(indicators[:15])}")
            for indicator in indicators[:12]:
                detailed_search_results.append(f"{indicator} ({bunya})")
            total_indicators += len(indicators)
        
        logger.info(f"📊 AIに提供する指標例: {len(bunya_groups)}分野, 総計{total_indicators}件")
        st.session_state['detailed_search_results'] = detailed_search_results
        
        return "\n".join(indicator_examples)
    except Exception as e:
        return f"指標リスト取得エラー: {str(e)}"

def generate_ai_analysis(query):
    """AIによる分析を実行し、推奨指標をJSON形式で返す"""
    logger.info(f"🤖 AI分析開始: '{query}'")
    available_indicators = get_available_indicators_for_query(query)
    system_prompt = f"""あなたは統計分析の専門家です。ユーザーの質問を分析し、複数の観点から関連する統計指標を推奨してください。
**重要**: 以下の実在する統計指標からのみ選択してください。存在しない指標は絶対に提案しないでください。
利用可能な統計指標：
{available_indicators}
出力は必ずJSON形式で、以下の構造に従ってください：
{{
  "analysis_perspectives": [
    {{
      "perspective_title": "分析観点のタイトル",
      "perspective_description": "この観点で分析する理由の説明",
      "recommended_indicators": [
        {{
          "indicator_name": "実在する統計指標名（上記リストから選択）",
          "recommendation_reason": "なぜこの指標を推奨するのかの理由"
        }}
      ]
    }}
  ]
}}
**必須要件**：
- indicator_nameは上記リストの実在する指標名と完全に一致させてください
- 各観点につき10個程度の指標を推奨してください
- 分析観点は4-5個に設定してください
- 総指標数は20-50個を目標にしてください
- JSON形式以外は出力しないでください
- 存在しない指標名は絶対に使用しないでください
- 上記リストから厳選して選択してください"""

    user_prompt = f"以下の質問について、統計分析の観点から多角的に分析し、関連する統計指標を推奨してください：\n\n{query}"
    
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
        st.error(f"AI分析でエラーが発生しました: {str(e)}")
        return None

def get_indicator_details(indicator_name):
    """指標名から、DataFrameに格納された詳細情報を取得する"""
    try:
        if retriever.df is None: return None
        
        row = None
        df = retriever.df
        
        exact_matches = df[df['koumoku_name_full'].str.strip() == indicator_name.strip()]
        if not exact_matches.empty: row = exact_matches.iloc[0]
        else:
            partial_matches = df[df['koumoku_name_full'].str.contains(indicator_name.strip(), na=False, case=False)]
            if not partial_matches.empty: row = partial_matches.iloc[0]

        if row is None: return None
        
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
        col_icon, col_content, col_actions = st.columns([0.3, 5.0, 0.7])

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
            
            # URL生成時に '#' を取り除く
            indicator_code = indicator_data.get("koumoku_code", "")
            cleaned_indicator_code = indicator_code.lstrip('#')
            power_bi_url = f"{base_url}&filter=social_demographic_pref_basic_bi/cat3_code eq '{cleaned_indicator_code}'"

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("📋", key=f"copy_{category_key}_{indicator_index}", help="Power BI URLをコピー"):
                    pyperclip.copy(power_bi_url)
                    st.toast("Power BI URLをコピーしました！", icon="📋")
            with action_col2:
                st.markdown(
                    f'<div style="text-align: center; padding-top: 5px;"><a href="{power_bi_url}" target="_blank" rel="noopener noreferrer" title="Power BIを新しいタブで開く">🔗</a></div>',
                    unsafe_allow_html=True
                )
        st.markdown('<hr style="margin: 4px 0; border: 0.5px solid #e0e0e0;">', unsafe_allow_html=True)

def display_ai_analysis_results(analysis_result, original_query):
    """AIによる分析結果全体を整形して表示する"""
    if not analysis_result or 'analysis_perspectives' not in analysis_result:
        st.error("分析結果の形式が正しくありません。")
        return

    st.markdown(
        f"""
        <div style="background-color: #e8f4fd; border-left: 4px solid #1f77b4; padding: 1rem; margin-bottom: 1rem;">
            「{original_query}」に関連して、以下の観点から指標をご提案します：
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for category_index, perspective in enumerate(analysis_result['analysis_perspectives']):
        valid_indicators = []
        category_key = f"category_{category_index}"
        
        for indicator in perspective.get('recommended_indicators', []):
            indicator_data = get_indicator_details(indicator.get('indicator_name'))
            if indicator_data:
                valid_indicators.append((indicator, indicator_data))
        
        if not valid_indicators: continue

        st.markdown("---")
        col_title, col_count = st.columns([4, 1])
        with col_title:
            st.markdown(f"## {perspective.get('perspective_title', '無題の観点')}")
            st.caption(perspective.get('perspective_description', ''))
        with col_count:
            st.markdown(f"**{len(valid_indicators)}件**")
        
        for indicator_index, (indicator, indicator_data) in enumerate(valid_indicators):
            display_indicator_card(
                indicator_data, 
                indicator.get('recommendation_reason', '理由なし'),
                category_key,
                indicator_index
            )
    
    if 'detailed_search_results' in st.session_state:
        st.markdown("---")
        with st.expander(f"🔍 参考：検索された全指標リスト ({len(st.session_state['detailed_search_results'])}件)"):
            for i, result in enumerate(st.session_state['detailed_search_results'], 1):
                st.write(f"  {i:2d}. {result}")
        del st.session_state['detailed_search_results']

def main():
    """アプリケーションのメインロジック"""
    st.title("社会・人口統計指標検索システム")

    st.markdown("""
        <style>
        .main > div { padding-top: 2rem; }
        .stTextInput > div > div > input { border-radius: 8px; border: 2px solid #e0e0e0; }
        .indicator-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin: 4px 0; background-color: #fafafa; transition: box-shadow 0.2s; }
        .indicator-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .indicator-title { font-size: 1.1em; font-weight: bold; color: #1f77b4; margin: 4px 0; }
        .indicator-code { color: #666; font-size: 0.9em; font-weight: bold; }
        .indicator-path { color: #888; font-size: 0.85em; margin: 2px 0; }
        .indicator-reason { color: #f39c12; font-size: 0.9em; margin: 4px 0; }
        </style>
    """, unsafe_allow_html=True)
    
    with st.spinner("📚 統計データベースを初期化中..."):
        if not retriever.load_vector_database():
            st.error("❌ データベースの読み込みに失敗しました")
            st.stop()
    
    available_models = llm_config.get_available_models()
    if available_models:
        col_model, _ = st.columns([2, 3])
        with col_model:
            model_options = list(available_models.keys())
            current_model_display = next((k for k, v in available_models.items() if v == llm_config.current_model), None)
            selected_model_display = st.selectbox(
                "🚀 AIモデル選択", model_options,
                index=model_options.index(current_model_display) if current_model_display else 0
            )
            selected_model = available_models[selected_model_display]
            if selected_model != llm_config.current_model:
                llm_config.set_model(selected_model)

    col_input, col_button = st.columns([4, 1])
    with col_input:
        query = st.text_input("質問を入力してください", placeholder="教育について知りたい", label_visibility="collapsed")
    with col_button:
        analyze_button = st.button("検索", type="primary", use_container_width=True)
    
    st.markdown(
        '<p style="color: #666; font-size: 0.9em; margin: 0px 0 0 -0.1;">例： 地域の教育水準を知りたい　高齢化の現状を把握したい　子育て環境を比較したい</p>',
        unsafe_allow_html=True
    )

    if analyze_button and query.strip():
        with st.spinner("🤖 AIが質問を分析し、最適な統計指標を検索中..."):
            with st.status("分析進行状況", expanded=True) as status:
                status.update(label="📝 質問を解釈しています...", state="running")
                time.sleep(0.5)
                status.update(label="🔍 関連指標を検索しています...", state="running")
                
                analysis_result = generate_ai_analysis(query.strip())
                
                if analysis_result:
                    status.update(label="📊 指標を整理しています...", state="running")
                    time.sleep(0.5)
                    st.session_state.analysis_result = analysis_result
                    st.session_state.original_query = query.strip()
                    status.update(label="✅ 分析完了!", state="complete")
                else:
                    status.update(label="❌ 分析失敗", state="error")
                    if 'analysis_result' in st.session_state:
                        del st.session_state.analysis_result
    
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        display_ai_analysis_results(
            st.session_state.analysis_result,
            st.session_state.original_query
        )
    elif analyze_button and not query.strip():
        st.warning("⚠️ 質問を入力してください。")

if __name__ == "__main__":
    st.set_page_config(
        page_title="社会・人口統計指標検索システム", 
        page_icon="",
        layout="wide"
    )
    main()