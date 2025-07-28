import streamlit as st
import json
import re
import pandas as pd
from retriever import retriever
from llm_config import llm_config
from loguru import logger
import time

# 対話の段階を定義
STAGE_INITIAL = "initial"
STAGE_PERSPECTIVE_SELECTION = "perspective_selection"
STAGE_GROUP_SELECTION = "group_selection"
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
    if 'selected_group_code' not in st.session_state:
        st.session_state.selected_group_code = None
    if 'selected_group_indicators' not in st.session_state:
        st.session_state.selected_group_indicators = []

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
    """分析観点を生成する（関連指標リストは不要）"""
    logger.info(f"🤖 分析観点生成開始: '{query}'")
    available_indicators = get_available_indicators_for_query(query)
    st.session_state.available_indicators = available_indicators
    
    system_prompt = f"""あなたは統計分析の専門家です。ユーザーの質問を分析し、4-5個の分析観点（中項目）を提示してください。

**重要**: 以下の実在する統計指標に基づいて観点を設定してください。
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
- 丁寧な『ですます調』にしてください
- JSON形式以外は出力しないでください"""

    user_prompt = f"以下の質問について、統計分析の観点から4-5個の分析観点を提示してください：\n\n{query}"
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.2)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            logger.info(f"✅ 分析観点を生成: {len(result.get('perspectives', []))}個の観点")
            return result
        else:
            logger.error(f"❌ 有効なJSONが生成されませんでした: {response[:500]}...")
            return None
    except Exception as e:
        logger.error(f"❌ 分析観点生成エラー: {str(e)}")
        st.error(f"分析観点生成でエラーが発生しました: {str(e)}")
        return None

def generate_dynamic_group_descriptions(user_query, group_list):
    """ユーザーのクエリに基づいて指標グループの動的説明文を生成する"""
    logger.info(f"🤖 動的説明文生成開始: '{user_query}' for {len(group_list)} groups")
    
    # グループ名リストを作成
    group_names = [group['title'] for group in group_list]
    
    system_prompt = f"""あなたは統計分析の専門家です。ユーザーのクエリに対して、各指標グループがどのような洞察を与えるかを説明してください。

**ユーザーのクエリ**: {user_query}

**指標グループリスト**: {', '.join(group_names)}

各指標グループについて、以下の観点から**より長く、丁寧な『ですます調』**で説明文を生成してください：
- このグループの指標を見ることで、ユーザーのクエリに対してどのような洞察が得られるか
- ユーザーがその指標グループを選ぶことで、どのようなことが分析できるのかを具体的に説明
- なぜこのグループがユーザーの関心事に関連するのか
- 1つの説明文は80-120文字程度で丁寧に

出力は必ずJSON形式で、以下の構造に従ってください：
[
  {{
    "group_name": "指標グループ名",
    "description": "ユーザーのクエリに対する洞察の詳細説明"
  }}
]

**必須要件**：
- group_nameは上記リストの指標グループ名と完全に一致させてください
- descriptionは80-120文字程度で、丁寧な「ですます調」で記述してください
- ユーザーにとって親切で分かりやすい説明にしてください
- JSON形式以外は出力しないでください"""

    user_prompt = f"上記の指標グループについて、ユーザーのクエリ「{user_query}」に対する洞察を説明してください。"
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.3)
        
        logger.info(f"🔍 LLM応答の最初の500文字: {response[:500]}")
        
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group())
            logger.info(f"✅ 動的説明文生成成功: {len(parsed_json)}件")
            return parsed_json
        else:
            logger.error(f"❌ 有効なJSONが生成されませんでした: {response[:500]}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析エラー: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ 動的説明文生成エラー: {str(e)}")
        return None

def generate_indicator_explanations(user_query, indicators_list):
    """各指標について、ユーザーのクエリとの関連性を説明する動的な文章を生成する"""
    logger.info(f"🤖 指標説明文生成開始: '{user_query}' for {len(indicators_list)} indicators")
    
    # 指標名リストを作成
    indicator_names = [indicator.get('koumoku_name_full', '') for indicator in indicators_list]
    
    system_prompt = f"""あなたは統計分析の専門家です。ユーザーのクエリに対して、各統計指標がなぜ重要なのかを説明してください。

**ユーザーのクエリ**: {user_query}

**統計指標リスト**: {', '.join(indicator_names)}

各統計指標について、以下の観点から簡潔で親切な説明文を生成してください：
- この指標がユーザーのクエリにどのように関連するのか
- この指標を見ることで何が分かるのか
- なぜこの指標が重要なのか
- 1つの説明文は60-80文字程度で「この指標は...」で始まる丁寧な文章

出力は必ずJSON形式で、以下の構造に従ってください：
{{
  "指標名1": "この指標は...",
  "指標名2": "この指標は...",
  ...
}}

**必須要件**：
- キーは上記リストの指標名と完全に一致させてください
- 値は60-80文字程度で「この指標は...」で始まる説明文にしてください
- ユーザーにとって分かりやすく親切な説明にしてください
- JSON形式以外は出力しないでください"""

    user_prompt = f"上記の統計指標について、ユーザーのクエリ「{user_query}」との関連性を説明してください。"
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.3)
        
        logger.info(f"🔍 指標説明文LLM応答の最初の500文字: {response[:500]}")
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group())
            logger.info(f"✅ 指標説明文生成成功: {len(parsed_json)}件")
            return parsed_json
        else:
            logger.error(f"❌ 有効なJSONが生成されませんでした: {response[:500]}")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析エラー: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"❌ 指標説明文生成エラー: {str(e)}")
        return {}

def generate_indicator_groups_for_perspective(perspective_title):
    """ステップA: hybrid_searchで関連指標を取得し、ステップB: group_codeで指標グループを生成"""
    logger.info(f"🤖 指標グループ生成開始: '{perspective_title}'")
    
    try:
        if retriever.df is None:
            retriever.load_vector_database()
        
        # ステップA: 分析観点のタイトルでhybrid_searchを実行し、関連指標を取得
        logger.info("📍 ステップA: hybrid_searchで関連指標を取得")
        search_results = retriever.hybrid_search(perspective_title, top_k=80)
        
        if not search_results:
            logger.error("❌ hybrid_searchで結果が取得できませんでした")
            return None
        
        # 検索結果から指標名リストを作成
        related_indicator_names = [result['koumoku_name_full'] for result in search_results]
        logger.info(f"📊 hybrid_searchで取得した関連指標数: {len(related_indicator_names)}件")
        
        # 一次フィルタリング済みDataFrameを作成
        df_filtered = retriever.df[retriever.df['koumoku_name_full'].isin(related_indicator_names)]
        logger.info(f"📋 一次フィルタリング済みDataFrame: {len(df_filtered)}行")
        
        if df_filtered.empty:
            logger.warning("⚠️ 一次フィルタリング済みDataFrameが空です")
            return None
        
        # ステップB: 一次フィルタリング済みDataFrameからユニークなgroup_codeを抽出
        logger.info("📍 ステップB: ユニークなgroup_codeを抽出")
        group_codes = df_filtered['group_code'].dropna().unique()
        logger.info(f"🔍 抽出されたグループコード: {group_codes.tolist()}")
        logger.info(f"🔍 グループコード数: {len(group_codes)}件")
        
        # 各グループの代表指標を取得
        group_indicators = []
        for group_code in sorted(group_codes):
            group_code_str = str(group_code)
            
            # 代表指標（group_codeと同じkoumoku_codeを持つ指標）を探す
            representative = retriever.df[retriever.df['koumoku_code'].astype(str) == group_code_str]
            
            if not representative.empty:
                row = representative.iloc[0]
                logger.info(f"✅ 代表指標: {group_code_str} -> {row['koumoku_name_full']}")
                group_indicators.append({
                    'group_code': group_code_str,
                    'title': row['koumoku_name_full'],
                    'description': f"「{row['koumoku_name_full']}」グループに含まれる全ての関連指標"
                })
            else:
                # フォールバック: そのgroup_codeを持つ任意の指標を代表とする
                fallback_indicators = df_filtered[df_filtered['group_code'].astype(str) == group_code_str]
                if not fallback_indicators.empty:
                    row = fallback_indicators.iloc[0]
                    logger.info(f"🔄 フォールバック代表指標: {group_code_str} -> {row['koumoku_name_full']}")
                    group_indicators.append({
                        'group_code': group_code_str,
                        'title': f"{row['koumoku_name_full']}関連グループ",
                        'description': f"「{row['koumoku_name_full']}」関連の指標グループ"
                    })
        
        # 上位15グループに制限
        group_indicators = group_indicators[:15]
        
        # 動的説明文を生成
        if group_indicators and st.session_state.original_query:
            dynamic_descriptions = generate_dynamic_group_descriptions(
                st.session_state.original_query, 
                group_indicators
            )
            
            if dynamic_descriptions:
                desc_map = {desc['group_name']: desc['description'] for desc in dynamic_descriptions}
                for group in group_indicators:
                    if group['title'] in desc_map:
                        group['description'] = desc_map[group['title']]
                        logger.info(f"✅ 動的説明文適用: {group['title']}")
        
        logger.info(f"✅ {len(group_indicators)}個の指標グループを生成")
        return {"groups": group_indicators}
        
    except Exception as e:
        logger.error(f"❌ 指標グループ生成エラー: {str(e)}")
        return None



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

def display_indicator_card(indicator_data, recommendation_reason, category_key, indicator_index, dynamic_explanation=None):
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
            
            # 動的説明文があれば表示、なければ従来の説明文を使用
            if dynamic_explanation:
                st.markdown(
                    f'<div class="indicator-reason">💡 {dynamic_explanation}</div>',
                    unsafe_allow_html=True
                )
            else:
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
    st.markdown("以下の選択肢から選択ボタンを押してください。")
    
    for i, option in enumerate(st.session_state.current_options):
        # 枠線で囲まれたカード
        with st.container(border=True):
            # 上段：タイトルと選択ボタン
            col_title, col_button = st.columns([4, 1])
            
            with col_title:
                st.markdown(f"**{i+1}. {option['title']}**")
            
            with col_button:
                if st.button("選択", key=f"perspective_{i}", type="primary", use_container_width=True):
                    st.session_state.selected_perspective = option
                    add_message_to_history("user", f"{i+1}番目の{option['title']}について詳しく知りたいです")
                    
                    # 選択された観点のタイトルで指標グループを生成
                    with st.spinner("🤖 関連する指標グループを抽出中..."):
                        groups_result = generate_indicator_groups_for_perspective(option['title'])
                        
                        if groups_result and 'groups' in groups_result and groups_result['groups']:
                            st.session_state.current_options = groups_result['groups']
                            st.session_state.stage = STAGE_GROUP_SELECTION
                            add_message_to_history("assistant", 
                                f"「{option['title']}」ですね。この観点に関連する指標グループを以下から選択してください。")
                        else:
                            logger.error(f"❌ 指標グループ生成結果が無効: {groups_result}")
                            st.error("関連する指標グループが見つかりませんでした。もう一度お試しください。")
                    st.rerun()
            
            # 下段：説明文
            st.markdown(option['description'])

def handle_group_selection_stage():
    """グループ選択段階：上位指標グループを提示してユーザーに選択してもらう"""
    perspective = st.session_state.selected_perspective
    st.markdown("### 📊 指標グループの選択")
    st.markdown(f"「{perspective['title']}」について、より具体的な指標グループを以下から選択してください。")
    
    if not st.session_state.current_options:
        st.error("指標グループデータが取得できませんでした。")
        if st.button("🔄 指標グループを再生成", key="regenerate_groups"):
            # 指標グループを再生成
            with st.spinner("🤖 指標グループを再生成中..."):
                groups_result = generate_indicator_groups_for_perspective(
                    perspective['title']
                )
                if groups_result and 'groups' in groups_result:
                    st.session_state.current_options = groups_result['groups']
                    st.rerun()
        return
    
    for i, group in enumerate(st.session_state.current_options):
        # 枠線で囲まれたカード
        with st.container(border=True):
            # 上段：タイトルと選択ボタン
            col_title, col_button = st.columns([4, 1])
            
            with col_title:
                st.markdown(f"**{i+1}. {group['title']} ({group['group_code']})**")
            
            with col_button:
                if st.button("選択", key=f"group_{i}", type="primary", use_container_width=True):
                    # 選択されたグループの全指標を取得
                    selected_group_code = group['group_code']
                    st.session_state.selected_group_code = selected_group_code
                    
                    # DataFrameから該当グループの全指標を取得
                    if retriever.df is not None:
                        group_indicators = retriever.df[
                            retriever.df['group_code'].astype(str) == str(selected_group_code)
                        ].copy()
                        
                        if not group_indicators.empty:
                            # セッション状態に保存
                            st.session_state.selected_group_indicators = group_indicators.to_dict('records')
                            st.session_state.stage = STAGE_FINAL
                            
                            add_message_to_history("user", f"{group['title']}グループの詳細が知りたい")
                            add_message_to_history("assistant", 
                                f"「{group['title']}」グループの指標を表示します。以下が関連する全ての指標です。")
                        else:
                            st.error("該当するグループの指標が見つかりませんでした。")
                    else:
                        st.error("データベースが読み込まれていません。")
                    st.rerun()
            
            # 下段：説明文
            st.markdown(group['description'])





def reset_session_state():
    """セッション状態をリセットして新しい検索を開始"""
    logger.info("🔄 セッション状態をリセット")
    for key in ['stage', 'current_options', 'selected_perspective', 'original_query', 'available_indicators', 'selected_group_code', 'selected_group_indicators']:
        if key in st.session_state:
            del st.session_state[key]
    
    # 初期状態に戻す
    st.session_state.stage = STAGE_INITIAL
    st.session_state.current_options = []

# check_if_new_query関数は不要になったため削除

def handle_final_stage():
    """最終段階：選択された指標グループの全件を表示"""
    st.markdown("### 📊 指標グループ詳細")
    
    if st.session_state.selected_group_indicators:
        # グループ情報の表示
        if st.session_state.selected_group_code:
            # 代表指標の情報を取得
            representative = None
            for indicator in st.session_state.selected_group_indicators:
                if indicator.get('koumoku_code') == st.session_state.selected_group_code:
                    representative = indicator
                    break
            
            if representative:
                st.markdown(f"**グループ**: {representative['koumoku_name_full']} ({st.session_state.selected_group_code})")
                st.markdown(f"**分野**: {representative['bunya_name']} > {representative['chuubunrui_name']} > {representative['shoubunrui_name']}")
        
        st.markdown(f"**該当指標数**: {len(st.session_state.selected_group_indicators)}件")
        st.markdown("---")
        
        # 動的説明文を生成
        indicator_explanations = {}
        if st.session_state.original_query:
            with st.spinner("🤖 各指標の詳細説明を生成中..."):
                indicator_explanations = generate_indicator_explanations(
                    st.session_state.original_query,
                    st.session_state.selected_group_indicators
                )
        
        # グループ内の全指標を表示
        for i, indicator_data in enumerate(st.session_state.selected_group_indicators):
            # 動的説明文を取得
            indicator_name = indicator_data.get('koumoku_name_full', '')
            dynamic_explanation = indicator_explanations.get(indicator_name, None)
            
            display_indicator_card(
                indicator_data, 
                f"「{st.session_state.selected_group_code}」グループに属する指標", 
                "group", 
                i,
                dynamic_explanation=dynamic_explanation
            )
    else:
        st.error("指標グループデータが見つかりません。")
    
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
        with st.spinner("🤖 分析観点を調査中..."):
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
    elif st.session_state.stage == STAGE_GROUP_SELECTION:
        handle_group_selection_stage()
    elif st.session_state.stage == STAGE_FINAL:
        handle_final_stage()
    
    # 会話履歴は表示しない

if __name__ == "__main__":
    main()