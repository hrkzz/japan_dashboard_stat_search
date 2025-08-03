import streamlit as st
import json
import re
import pandas as pd
import uuid
from retriever import retriever
from llm_config import llm_config
from loguru import logger
import time
try:
    from bq_logger import log_event
    LOGGING_ENABLED = True
except ImportError as e:
    logger.warning(f"⚠️ BigQueryロガーのインポートに失敗: {str(e)}")
    LOGGING_ENABLED = False
    def log_event(*args, **kwargs):
        """ダミーのlog_event関数"""
        return False

# 対話の段階を定義
STAGE_INITIAL = "initial"
STAGE_PERSPECTIVE_SELECTION = "perspective_selection"
STAGE_GROUP_SELECTION = "group_selection"
STAGE_FINAL = "final"

def initialize_session_state():
    """セッション状態を初期化"""
    # セッションIDの生成（初回のみ）
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"🆔 新しいセッションID生成: {st.session_state.session_id[:8]}...")
    
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
    if 'analysis_plan' not in st.session_state:
        st.session_state.analysis_plan = None

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

@st.cache_data(ttl=3600, show_spinner=False)
def generate_analysis_plan(query):
    """
    ユーザーのクエリに基づき、分析観点とそれに紐づく指標グループ案を含む
    「分析計画」を一度のLLMコールで生成する。
    """
    logger.info(f"🤖 分析計画の生成開始: '{query}'")
    available_indicators = get_available_indicators_for_query(query)
    st.session_state.available_indicators = available_indicators

    system_prompt = f"""あなたは優秀なデータ分析の専門家です。ユーザーの質問を分析し、複数の「分析観点」と、各観点を探るための具体的な「指標グループ案」を提案してください。

# 指示
ユーザーの質問に対し、包括的な分析計画をJSON形式で出力してください。
分析計画には、8〜10個の「分析観点（perspectives）」を含めてください。
各「分析観点」には、その観点で具体的に何を見るべきかを示す「指標グループ案（suggested_groups）」を2〜3個含めてください。

# 利用可能な統計指標の例
{available_indicators}

# 出力形式（必ずこのJSON構造に従うこと）
{{
  "analysis_plan": {{
    "perspectives": [
      {{
        "perspective_title": "（例）教育環境の充実度",
        "perspective_description": "地域の教育水準や子育て世代への教育支援がどの程度手厚いかを評価するための観点です。",
        "suggested_groups": [
          {{
            "group_title": "（例）学校教育と施設",
            "group_description": "公立学校の数、教員一人当たりの生徒数、図書館や体育館といった施設の状況から、基礎的な教育環境の質を把握します。"
          }},
          {{
            "group_title": "（例）保育・待機児童問題",
            "group_description": "保育所の数や待機児童の状況です。これが改善されれば、共働き世帯が安心して子育てできる環境が整っていると言えます。"
          }}
        ]
      }}
    ]
  }}
}}

# 制約
- JSON形式以外は絶対に出力しないでください。
- 説明文は丁寧な「ですます調」で、ユーザーが理解しやすいように記述してください。
"""
    user_prompt = f"以下の質問について、詳細な分析計画をJSON形式で提案してください：\n\n{query}"

    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.2)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # 分析観点を最大10個に制限
            if 'analysis_plan' in result and 'perspectives' in result['analysis_plan']:
                result['analysis_plan']['perspectives'] = result['analysis_plan']['perspectives'][:10]
            logger.info("✅ 分析計画の生成に成功しました。")
            return result
        else:
            logger.error(f"❌ 分析計画のJSON生成に失敗: {response[:500]}...")
            return None
    except Exception as e:
        logger.error(f"❌ 分析計画の生成エラー: {str(e)}")
        st.error(f"分析計画の生成でエラーが発生しました: {str(e)}")
        return None



def generate_group_summary(group_indicators, user_query):
    """選択されたグループの要約文をAIで生成する"""
    logger.info(f"🤖 グループ要約生成開始: '{user_query}' for {len(group_indicators)} indicators")
    
    # 指標名リストを作成
    indicator_names = [indicator.get('koumoku_name_full', '') for indicator in group_indicators]
    
    # 分野情報を取得
    representative = group_indicators[0] if group_indicators else {}
    bunya_info = f"{representative.get('bunya_name', '')} > {representative.get('chuubunrui_name', '')} > {representative.get('shoubunrui_name', '')}"
    
    system_prompt = f"""あなたは統計分析の専門家です。ユーザーのクエリに対して、選択された指標グループがどのような分析の切り口を提供するかを説明してください。

**ユーザーのクエリ**: {user_query}

**指標グループに含まれる指標**: {', '.join(indicator_names[:10])}{'...' if len(indicator_names) > 10 else ''}

**分野**: {bunya_info}

**指標数**: {len(group_indicators)}件

このグループの指標について、以下の観点から簡潔で分かりやすい要約文を生成してください：
- ユーザーの関心事に対し、このグループがどのような分析の切り口を提供するか
- このグループの指標を見ることで何がわかるのか
- なぜこのグループがユーザーの関心事に重要なのか
- 要約文は150-250文字程度で、丁寧な「ですます調」で記述

出力は要約文のみを出力してください。JSON形式や他の形式は不要です。"""

    user_prompt = f"上記の指標グループについて、ユーザーのクエリ「{user_query}」に対する要約説明文を生成してください。"
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_config.generate_response(messages, temperature=0.3)
        
        logger.info(f"✅ グループ要約生成成功")
        return response.strip()
    except Exception as e:
        logger.error(f"❌ グループ要約生成エラー: {str(e)}")
        return "このグループの指標は、ユーザーの関心事に関連する重要な統計データを含んでいます。"

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
            
            # 冗長な説明文を削除してカードをシンプル化
            # if dynamic_explanation:
            #     st.markdown(
            #         f'<div class="indicator-reason">💡 {dynamic_explanation}</div>',
            #         unsafe_allow_html=True
            #     )
            # else:
            #     st.markdown(
            #         f'<div class="indicator-reason">💡 {recommendation_reason}</div>',
            #         unsafe_allow_html=True
            #     )
            
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
    st.markdown("### 統計指標検索アシスタント")
    st.markdown("どのような統計指標をお探しですか？分析したいテーマを下のチャット欄に入力してください。")
    st.markdown("例: 子育て環境を比較したい、地域の教育水準を知りたい、高齢化の現状を把握したい")

def handle_perspective_selection_stage():
    """観点選択段階：分析観点を提示してユーザーに選択してもらう"""
    st.markdown("### 分析観点の選択")
    st.markdown(f"「{st.session_state.original_query}」について、どのような観点で分析しますか？")
    st.markdown("以下の選択肢から選択ボタンを押してください。")
    
    perspectives = st.session_state.analysis_plan.get('perspectives', [])
    for i, perspective in enumerate(perspectives):
        # 枠線で囲まれたカード
        with st.container(border=True):
            # 上段：タイトルと選択ボタン
            col_title, col_button = st.columns([4, 1])
            
            with col_title:
                st.markdown(f"**{i+1}. {perspective['perspective_title']}**")
            
            with col_button:
                if st.button("選択", key=f"perspective_{i}", type="primary", use_container_width=True):
                    st.session_state.selected_perspective = perspective
                    add_message_to_history("user", f"{i+1}番目の{perspective['perspective_title']}について詳しく知りたいです")
                    
                    # analysis_planからsuggested_groupsを取得
                    st.session_state.current_options = perspective.get('suggested_groups', [])
                    st.session_state.stage = STAGE_GROUP_SELECTION
                    add_message_to_history("assistant", 
                        f"「{perspective['perspective_title']}」ですね。この観点に関連する指標グループを以下から選択してください。")
                    st.rerun()
            
            # 下段：説明文
            st.markdown(perspective['perspective_description'])

def handle_group_selection_stage():
    """グループ選択段階：上位指標グループを提示してユーザーに選択してもらう"""
    perspective = st.session_state.selected_perspective
    st.markdown("### 📊 指標グループの選択")
    st.markdown(f"「{perspective['perspective_title']}」について、より具体的な指標グループを以下から選択してください。")
    
    if not st.session_state.current_options:
        st.error("指標グループデータが取得できませんでした。")
        return
    
    for i, group in enumerate(st.session_state.current_options):
        # 枠線で囲まれたカード
        with st.container(border=True):
            # 上段：タイトルと選択ボタン
            col_title, col_button = st.columns([4, 1])
            
            with col_title:
                st.markdown(f"**{i+1}. {group['group_title']}**")
            
            with col_button:
                if st.button("選択", key=f"group_{i}", type="primary", use_container_width=True):
                    # ユーザーが選択したグループ案のタイトルを取得
                    selected_group_title = group['group_title']
                    add_message_to_history("user", f"「{selected_group_title}」グループの詳細が知りたい")

                    with st.spinner(f"「{selected_group_title}」グループの指標を検索・集計中..."):
                        # Part 1で修正したhybrid_searchを使い、代表的な指標グループを検索
                        search_results = retriever.hybrid_search(selected_group_title, top_k=5)

                        if not search_results:
                            st.error("関連する指標グループが見つかりませんでした。")
                            st.stop()

                        # 最も関連性の高い代表指標のkoumoku_codeを特定
                        selected_koumoku_code = search_results[0]['koumoku_code']
                        logger.info(f"特定された代表指標コード: {selected_koumoku_code}")

                        # retrieverが持つ完全なデータフレームから、
                        # 代表指標のコードで前方一致する指標のみを抽出する
                        if retriever.df is not None:
                            # .astype(str) でデータ型を揃えてから前方一致検索
                            group_indicators_df = retriever.df[
                                retriever.df['koumoku_code'].astype(str).str.startswith(str(selected_koumoku_code))
                            ].copy()

                            if not group_indicators_df.empty:
                                # セッション状態に「選択された1グループの指標のみ」を保存
                                st.session_state.selected_group_indicators = group_indicators_df.to_dict('records')
                                st.session_state.selected_group_code = selected_koumoku_code  # 完全なコードを保存
                                st.session_state.stage = STAGE_FINAL
                                
                                representative_name = group_indicators_df.iloc[0]['koumoku_name_full']
                                
                                # 選択項目のロギング
                                if LOGGING_ENABLED:
                                    try:
                                        current_model = getattr(llm_config, 'current_model', 'unknown')
                                        selected_perspective_title = st.session_state.selected_perspective.get('perspective_title', '') if st.session_state.selected_perspective else ''
                                        
                                        log_event(
                                            session_id=st.session_state.session_id,
                                            event_type='selection',
                                            user_query=st.session_state.original_query,
                                            selected_perspective=selected_perspective_title,
                                            selected_group=selected_group_title,
                                            final_indicators=st.session_state.selected_group_indicators,
                                            llm_model=current_model
                                        )
                                    except Exception as e:
                                        logger.warning(f"⚠️ 選択ログ記録エラー（機能は継続します）: {str(e)}")
                                
                                add_message_to_history("assistant", 
                                    f"承知いたしました。「{representative_name}」に関連する指標を表示します。")
                            else:
                                st.error(f"指標グループ({selected_koumoku_code})に属する指標が見つかりませんでした。")
                        else:
                            st.error("データベースが読み込まれていません。")
                    
                    st.rerun()
            
            # 下段：説明文
            st.markdown(group['group_description'])

def reset_session_state():
    """セッション状態をリセットして新しい検索を開始"""
    logger.info("🔄 セッション状態をリセット")
    for key in ['stage', 'current_options', 'selected_perspective', 'original_query', 'available_indicators', 'selected_group_code', 'selected_group_indicators', 'analysis_plan', 'session_id']:
        if key in st.session_state:
            del st.session_state[key]
    
    # 初期状態に戻す
    st.session_state.stage = STAGE_INITIAL
    st.session_state.current_options = []
    # 新しいセッションIDを生成
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"🆔 新しいセッションID生成: {st.session_state.session_id[:8]}...")

def handle_final_stage():
    """最終段階：選択された指標グループの全件を表示"""
    st.markdown("### 📊 指標グループ詳細")
    
    if st.session_state.selected_group_indicators:
        # グループ情報の表示
        representative = st.session_state.selected_group_indicators[0] if st.session_state.selected_group_indicators else None
        
        if representative:
            st.markdown(f"**グループ**: {representative['koumoku_name_full']}関連指標")
            st.markdown(f"**分野**: {representative['bunya_name']} > {representative['chuubunrui_name']} > {representative['shoubunrui_name']}")
        
        st.markdown(f"**該当指標数**: {len(st.session_state.selected_group_indicators)}件")
        
        # グループ要約を生成・表示
        if st.session_state.original_query:
            with st.spinner("グループ要約を生成中..."):
                group_summary = generate_group_summary(
                    st.session_state.selected_group_indicators,
                    st.session_state.original_query
                )
            
            # 区切り線の追加とクリーンな概要表示
            st.divider()
            st.markdown(group_summary)
            st.divider()
        
        # 指標件数に応じた表示処理の分岐
        total_indicators = len(st.session_state.selected_group_indicators)
        
        if total_indicators > 1:
            # 複数指標の場合：親子指標の分離（代表指標自身を一覧から除外）
            child_indicators = []
            representative_code = st.session_state.get('selected_group_code', '')
            
            for indicator_data in st.session_state.selected_group_indicators:
                # 代表指標自身（完全一致）を除外
                if str(indicator_data.get('koumoku_code', '')) != str(representative_code):
                    child_indicators.append(indicator_data)
            
            # 子指標のみを表示
            st.markdown(f"### 📊 関連指標一覧（{len(child_indicators)}件）")
            for i, indicator_data in enumerate(child_indicators):
                display_indicator_card(
                    indicator_data, 
                    "前方一致による関連指標", 
                    "group", 
                    i
                )
        else:
            # 単独指標の場合：タイトルなしでその1件をそのまま表示
            for i, indicator_data in enumerate(st.session_state.selected_group_indicators):
                display_indicator_card(
                    indicator_data, 
                    "代表指標", 
                    "group", 
                    i
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
        # クエリのロギング
        if LOGGING_ENABLED:
            try:
                current_model = getattr(llm_config, 'current_model', 'unknown')
                log_event(
                    session_id=st.session_state.session_id,
                    event_type='query',
                    user_query=user_input,
                    llm_model=current_model
                )
            except Exception as e:
                logger.warning(f"⚠️ クエリログ記録エラー（機能は継続します）: {str(e)}")
        
        # 初期段階：分析計画を生成
        with st.spinner("分析計画を調査中..."):
            plan_result = generate_analysis_plan(user_input)
            
            if plan_result and 'analysis_plan' in plan_result:
                st.session_state.analysis_plan = plan_result['analysis_plan']
                st.session_state.original_query = user_input
                st.session_state.stage = STAGE_PERSPECTIVE_SELECTION
                add_message_to_history("assistant", f"承知いたしました。「{user_input}」についてですね。どのような観点で分析しますか？")
            else:
                add_message_to_history("assistant", "申し訳ございません。分析計画の生成に失敗しました。もう一度お試しください。")

def main():
    """アプリケーションのメインロジック"""
    st.set_page_config(
        page_title="統計指標検索アシスタント", 
        page_icon="",
        layout="wide"
    )
    
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

if __name__ == "__main__":
    main()