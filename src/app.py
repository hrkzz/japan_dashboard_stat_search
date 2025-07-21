import streamlit as st
import json
import re
from retriever import retriever
from llm_config import llm_config
from loguru import logger

def get_available_indicators_for_query(query):
    """クエリに関連する指標を動的に取得して豊富なリストを提供"""
    try:
        if retriever.df is None:
            retriever.load_vector_database()
        
        logger.info(f"🔍 指標例取得開始: '{query}'")
        
        # クエリに基づく検索結果を取得（適量）
        search_results = retriever.hybrid_search(query, top_k=40)
        
        # 分野別にグループ化
        bunya_groups = {}
        for result in search_results:
            bunya = result['bunya_name'] 
            if bunya not in bunya_groups:
                bunya_groups[bunya] = []
            bunya_groups[bunya].append(result['koumoku_name_full'])
        
        # さらに分野別に追加の指標を補完（高速化のため削減）
        for bunya in bunya_groups.keys():
            bunya_indicators = retriever.df[retriever.df['bunya_name'] == bunya]['koumoku_name_full'].tolist()
            # 既存の指標に加えて、その分野の他の指標も追加（高速化のため10個に削減）
            existing = set(bunya_groups[bunya])
            additional = [ind for ind in bunya_indicators if ind not in existing][:10]
            bunya_groups[bunya].extend(additional)
        
        # 結果を整形
        indicator_examples = []
        detailed_search_results = []
        total_indicators = 0
        for bunya, indicators in bunya_groups.items():
            # 各分野で最大15個の指標例を提供（高速化のため削減）
            indicator_examples.append(f"【{bunya}】({len(indicators)}件利用可能): {', '.join(indicators[:15])}")
            
            # 詳細リスト用（高速化のため12個に削減）
            for indicator in indicators[:12]:
                detailed_search_results.append(f"{indicator} ({bunya})")
            
            total_indicators += len(indicators)
        
        logger.info(f"📊 AIに提供する指標例: {len(bunya_groups)}分野, 総計{total_indicators}件")
        
        # グローバル変数に詳細結果を保存（AI分析後に表示用）
        st.session_state['detailed_search_results'] = detailed_search_results
        
        return "\n".join(indicator_examples)
    except Exception as e:
        return f"指標リスト取得エラー: {str(e)}"

def generate_ai_analysis(query):
    """AIによるクエリ分析と指標推奨"""
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm_config.generate_response(messages, temperature=0.2)
        logger.info(f"🤖 AI元レスポンス（最初の500文字）: {response[:500]}")
        logger.info(f"🤖 AI元レスポンス（最後の200文字）: {response[-200:]}")
        
        # JSONパース
        try:
            analysis_result = json.loads(response)
            logger.info(f"🔧 JSON直接パース成功: {type(analysis_result)}")
            logger.info(f"🔧 分析結果キー: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
            
            # AI分析結果をログ出力
            if analysis_result and 'analysis_perspectives' in analysis_result:
                total_recommended = 0
                for i, perspective in enumerate(analysis_result['analysis_perspectives']):
                    count = len(perspective.get('recommended_indicators', []))
                    total_recommended += count
                    logger.info(f"📋 観点{i+1}: '{perspective.get('perspective_title', 'Unknown')}' - {count}個の指標を推奨")
                logger.info(f"🎯 AI推奨指標総数: {total_recommended}件")
            else:
                logger.warning(f"⚠️ analysis_perspectives キーが見つかりません: {analysis_result}")
            
            return analysis_result
        except json.JSONDecodeError as e:
            logger.debug(f"📋 Markdownコードブロック形式を検出: {str(e)}")
            # AIがMarkdownコードブロック形式（```json...```）で返した場合の処理
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    parsed_result = json.loads(json_match.group())
                    logger.info("✅ JSON抽出パースで成功")
                    logger.debug(f"🔧 抽出結果キー: {list(parsed_result.keys()) if isinstance(parsed_result, dict) else 'Not a dict'}")
                    
                    if parsed_result and 'analysis_perspectives' in parsed_result:
                        return parsed_result
                    else:
                        logger.warning(f"⚠️ 抽出結果にanalysis_perspectivesキーなし: {parsed_result}")
                        return parsed_result
                except json.JSONDecodeError as extract_error:
                    logger.error(f"❌ JSON抽出パースもエラー: {str(extract_error)}")
                    logger.error(f"❌ 元のレスポンス: {response[:500]}...")
                    return None
            else:
                logger.error("❌ 有効なJSONが生成されませんでした")
                logger.error(f"❌ 元のレスポンス: {response[:500]}...")
                raise ValueError("有効なJSONが生成されませんでした")
                
    except Exception as e:
        st.error(f"AI分析でエラーが発生しました: {str(e)}")
        return None

def get_indicator_details(indicator_name):
    """指標名から詳細情報を取得（改良版マッチング）"""
    try:
        if retriever.df is None:
            return None
        
        # 複数の方法でマッチングを試行
        row = None
        
        # 方法1: 完全一致
        exact_matches = retriever.df[
            retriever.df['koumoku_name_full'].str.strip() == indicator_name.strip()
        ]
        if not exact_matches.empty:
            row = exact_matches.iloc[0]
        
        # 方法2: 前方一致
        if row is None:
            prefix_matches = retriever.df[
                retriever.df['koumoku_name_full'].str.startswith(indicator_name.strip(), na=False)
            ]
            if not prefix_matches.empty:
                row = prefix_matches.iloc[0]
        
        # 方法3: 部分一致
        if row is None:
            partial_matches = retriever.df[
                retriever.df['koumoku_name_full'].str.contains(indicator_name.strip(), na=False, case=False)
            ]
            if not partial_matches.empty:
                row = partial_matches.iloc[0]
        
        # 方法4: koumoku_nameでの検索（存在する場合）
        if row is None and 'koumoku_name' in retriever.df.columns:
            name_matches = retriever.df[
                retriever.df['koumoku_name'].str.contains(indicator_name.strip(), na=False, case=False)
            ]
            if not name_matches.empty:
                row = name_matches.iloc[0]
        
        if row is None:
            return None
        
        # 結果を整形
        result = {
            'koumoku_name_full': row['koumoku_name_full'],
            'bunya_name': row['bunya_name'],
            'chuubunrui_name': row['chuubunrui_name'],
            'shoubunrui_name': row['shoubunrui_name'],
        }
        
        # オプション項目の安全な取得
        if 'koumoku_name' in retriever.df.columns:
            result['koumoku_name'] = row.get('koumoku_name', row['koumoku_name_full'])
        else:
            result['koumoku_name'] = row['koumoku_name_full']
            
        if 'stat_name' in retriever.df.columns:
            result['stat_name'] = row.get('stat_name', '')
        else:
            result['stat_name'] = ''
            
        if 'koumoku_code' in retriever.df.columns:
            result['koumoku_code'] = row.get('koumoku_code', '')
        else:
            result['koumoku_code'] = str(row.name)
        
        return result
        
    except Exception as e:
        st.error(f"指標詳細取得エラー: {str(e)}")
        return None

def display_indicator_card(indicator_data, recommendation_reason, category_key, indicator_index):
    """指標カードを表示（新デザイン）"""
    if not indicator_data:
        st.error("指標データが無効です")
        return
    
    # カードスタイル定義
    card_css = """
    <style>
    .indicator-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        background-color: #fafafa;
        transition: box-shadow 0.2s;
    }
    .indicator-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .indicator-title {
        font-size: 1.1em;
        font-weight: bold;
        color: #1f77b4;
        margin: 4px 0;
    }
    .indicator-code {
        color: #666;
        font-size: 0.9em;
        font-weight: bold;
    }
    .indicator-path {
        color: #888;
        font-size: 0.85em;
        margin: 2px 0;
    }
    .indicator-reason {
        color: #f39c12;
        font-size: 0.9em;
        margin: 4px 0;
    }

    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)
    
    with st.container():
        col_icon, col_content, col_actions = st.columns([0.3, 4.5, 1])
        
        with col_icon:
            st.markdown("📊")
        
        with col_content:
            # 指標名とコード
            st.markdown(
                f'<div class="indicator-title">{indicator_data["koumoku_name_full"]} '
                f'<span class="indicator-code">{indicator_data.get("koumoku_code", "")}</span></div>',
                unsafe_allow_html=True
            )
            
            # 推奨理由（オレンジの💡アイコン付き）
            st.markdown(
                f'<div class="indicator-reason">💡 {recommendation_reason}</div>',
                unsafe_allow_html=True
            )
            
            # 階層パス
            path = f'{indicator_data["bunya_name"]} > {indicator_data["chuubunrui_name"]} > {indicator_data["shoubunrui_name"]}'
            st.markdown(
                f'<div class="indicator-path">{path}</div>',
                unsafe_allow_html=True
            )
            

        
        with col_actions:
            # アクションボタン
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("📋", key=f"copy_{category_key}_{indicator_index}", help="コピー"):
                    st.success("コピーしました", icon="✅")
            with action_col2:
                if st.button("🔗", key=f"link_{category_key}_{indicator_index}", help="外部リンク"):
                    st.info("外部リンク機能は今後実装予定です")
        
        # カード区切り線
        st.markdown('<hr style="margin: 4px 0; border: 0.5px solid #e0e0e0;">', unsafe_allow_html=True)

def display_ai_analysis_results(analysis_result, original_query):
    """AI分析結果を表示（新デザイン）"""
    logger.info(f"🖥️ 表示関数に渡されたデータ型: {type(analysis_result)}")
    logger.info(f"🖥️ 表示関数に渡されたデータ: {analysis_result}")
    
    if not analysis_result:
        st.error("分析結果が空です。")
        logger.error("❌ 分析結果が空です")
        return
        
    if 'analysis_perspectives' not in analysis_result:
        st.error(f"分析結果の形式が正しくありません。受信キー: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
        logger.error(f"❌ analysis_perspectivesキーが見つかりません。利用可能キー: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
        return
    
    # 検索結果ヘッダー（青いフィールドで表示）
    st.markdown(
        f"""
        <div style="background-color: #e8f4fd; border-left: 4px solid #1f77b4; padding: 1rem; margin-bottom: 1rem;">
            「{original_query}」に関連して、以下の観点から指標をご提案します：
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for category_index, perspective in enumerate(analysis_result['analysis_perspectives']):
        # 指標詳細を取得して有効な指標のみ抽出
        valid_indicators = []
        category_key = f"category_{category_index}"
        
        for indicator in perspective['recommended_indicators']:
            indicator_data = get_indicator_details(indicator['indicator_name'])
            if indicator_data:
                valid_indicators.append((indicator, indicator_data))
        
        # 有効な指標がない観点はスキップ
        if not valid_indicators:
            continue
        
        # セクション区切り線
        st.markdown("---")
        
        # カテゴリヘッダー
        col_title, col_count = st.columns([4, 1])
        with col_title:
            st.markdown(f"## {perspective['perspective_title']}")
            st.caption(perspective['perspective_description'])
        with col_count:
            st.markdown(f"**{len(valid_indicators)}件**")
        
        # 指標カードを表示（全件表示）
        for indicator_index, (indicator, indicator_data) in enumerate(valid_indicators):
            display_indicator_card(
                indicator_data, 
                indicator['recommendation_reason'],
                category_key,
                indicator_index
            )
        
        # 追加のスペース
        st.markdown("")
    
    # AI分析後に検索結果詳細を表示
    if 'detailed_search_results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 参考：検索された全指標リスト")
        st.markdown("*AIが分析に使用した指標の詳細一覧です*")
        
        with st.expander(f"🔍 検索結果詳細 ({len(st.session_state['detailed_search_results'])}件)", expanded=False):
            for i, result in enumerate(st.session_state['detailed_search_results'], 1):
                st.write(f"  {i:2d}. {result}")
        
        # セッションステートをクリア
        del st.session_state['detailed_search_results']
    
    # 最終的な情報メッセージ（削除）
    st.markdown("---")

def main():
    
    # ヘッダー
    st.title("社会・人口統計指標検索システム")
    
    # シンプルなスタイリング
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # データベース初期化
    with st.spinner("📚 統計データベースを初期化中..."):
        if not retriever.load_vector_database():
            st.error("❌ データベースの読み込みに失敗しました")
            st.stop()
    
    # モデル選択機能
    available_models = llm_config.get_available_models()
    if available_models:
        col_model, col_spacer = st.columns([2, 3])
        with col_model:
            model_options = list(available_models.keys())
            current_model_display = None
            for display_name, model_name in available_models.items():
                if model_name == llm_config.current_model:
                    current_model_display = display_name
                    break
            
            selected_model_display = st.selectbox(
                "🚀 AIモデル選択",
                model_options,
                index=model_options.index(current_model_display) if current_model_display else 0
            )
            
            # 選択されたモデルを設定
            selected_model = available_models[selected_model_display]
            if selected_model != llm_config.current_model:
                llm_config.set_model(selected_model)

    # メイン検索インターフェース
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        query = st.text_input(
            "質問を入力してください",
            placeholder="教育について知りたい",
            label_visibility="collapsed"
        )
    
    with col_button:
        analyze_button = st.button("検索", type="primary", use_container_width=True)
    
    # サンプル質問
    st.markdown(
        '<p style="color: #666; font-size: 0.9em; margin: 0px 0 0 -0.1;">例： '
        '<span style="color: ##666;">地域の教育水準を知りたい　</span> '
        '<span style="color: ##666;">高齢化の現状を把握したい　</span> '
        '<span style="color: ##666;">子育て環境を比較したい　</span> '
        '<span style="color: ##666;">医療体制の充実度を調べたい　</span></p>',
        unsafe_allow_html=True
        )
    
    # AI分析・結果表示
    if analyze_button and query.strip():
        with st.spinner("🤖 AIが質問を分析し、最適な統計指標を検索中..."):
            # ステップ1: AI分析
            with st.status("分析進行状況", expanded=True) as status:
                # ステップ1: 質問解釈
                step1 = st.empty()
                step2 = st.empty()
                step3 = st.empty()
                step4 = st.empty()
                
                step1.markdown("📝 質問を解釈しています...")
                step2.markdown("<span style='color:#ccc'>🔍 関連指標を検索しています...</span>", unsafe_allow_html=True)
                step3.markdown("<span style='color:#ccc'>📊 指標を整理しています...</span>", unsafe_allow_html=True)
                step4.markdown("<span style='color:#ccc'>✅ 分析観点を特定しました</span>", unsafe_allow_html=True)
                
                import time
                time.sleep(0.5)
                
                # ステップ2: 指標検索開始
                step1.markdown("✅ 質問を解釈しています...")
                step2.markdown("🔍 関連指標を検索しています...")
                step3.markdown("<span style='color:#ccc'>📊 指標を整理しています...</span>", unsafe_allow_html=True)
                step4.markdown("<span style='color:#ccc'>✅ 分析観点を特定しました</span>", unsafe_allow_html=True)
                
                time.sleep(0.3)
                
                # ステップ3: AI分析実行
                step2.markdown("✅ 関連指標を検索しています...")
                step3.markdown("📊 指標を整理しています...")
                
                analysis_result = generate_ai_analysis(query.strip())
                
                if analysis_result:
                    # ステップ4: 完了
                    step2.markdown("✅ 関連指標を検索しています...")
                    step3.markdown("✅ 指標を整理しています...")
                    step4.markdown("✅ 分析観点を特定しました")
                    status.update(label="✅ 分析完了!", state="complete")
                else:
                    step4.markdown("❌ 分析に失敗しました")
                    status.update(label="❌ 分析失敗", state="error")
                    st.stop()
        
        # ステップ2: 結果表示
        st.markdown("---")
        
        # AI分析結果の表示
        display_ai_analysis_results(analysis_result, query)
        
    elif analyze_button and not query.strip():
        st.warning("⚠️ 質問を入力してください。")

if __name__ == "__main__":
    # ページ設定（最初に実行する必要がある）
    st.set_page_config(
        page_title="社会・人口統計指標検索システム", 
        page_icon="🤖",
        layout="wide"
    )
    main() 