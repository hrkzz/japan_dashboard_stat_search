from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from loguru import logger

from state_manager import (
    STAGE_FINAL,
    STAGE_GROUP_SELECTION,
    STAGE_INITIAL,
    STAGE_PERSPECTIVE_SELECTION,
    StateManager,
)
from services import AnalysisService

try:
    from bq_logger import log_event  # type: ignore
    LOGGING_ENABLED = True
except Exception:
    LOGGING_ENABLED = False
    def log_event(**kwargs):  # type: ignore
        return False

try:
    from streamlit_extras.st_javascript import st_javascript  # type: ignore
    JAVASCRIPT_ENABLED = True
except Exception:
    JAVASCRIPT_ENABLED = False
    def st_javascript(js_code):  # type: ignore
        return None

try:
    import pyperclip  # type: ignore
    PYPERCLIP_AVAILABLE = True
except Exception:
    PYPERCLIP_AVAILABLE = False

from llm_config import llm_config


def display_indicator_card(state: StateManager, indicator_data: Dict[str, Any], category_key: str, indicator_index: int) -> None:
    if not indicator_data:
        st.error("指標データが無効です")
        return

    with st.container(border=True):
        col_icon, col_content = st.columns([0.5, 9.5])
        with col_icon:
            st.markdown("📊")
        with col_content:
            indicator_code = indicator_data.get("koumoku_code", "")
            st.markdown(
                f'<div class="indicator-title">{indicator_data["koumoku_name_full"]} '
                f'<span class="indicator-code">{indicator_code.lstrip("#")}</span></div>',
                unsafe_allow_html=True,
            )
            path = (
                f'{indicator_data["bunya_name"]} > {indicator_data["chuubunrui_name"]} > {indicator_data["shoubunrui_name"]}'
            )
            st.markdown(
                f'<div class="indicator-path">{path}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="margin: 4px 0; border: 0.5px solid #e0e0e0;">', unsafe_allow_html=True)
        action_col1, action_col2 = st.columns(2)

        with action_col1:
            unique_key = f"add_{category_key}_{indicator_index}_{indicator_data.get('koumoku_code', '')}"
            if st.button("✔ 保存リストへ", key=unique_key, type="primary", use_container_width=True):
                koumoku_code = indicator_data.get("koumoku_code", "")
                is_duplicate = any(
                    saved.get("koumoku_code", "") == koumoku_code for saved in state.get_saved_indicators()
                )
                if not is_duplicate:
                    state.add_saved_indicator(indicator_data)
                    if LOGGING_ENABLED:
                        try:
                            log_event(
                                session_id=state.get_session_id(),
                                event_type="add_indicator",
                                user_query=state.get_original_query(),
                                selected_indicator=indicator_data,
                                selected_perspective=(state.get_selected_perspective() or {}).get(
                                    "perspective_title", ""
                                ),
                                selected_group=st.session_state.get("selected_group_title", ""),
                                final_indicators=state.get_selected_group_indicators(),
                                llm_model=getattr(llm_config, "current_model", "unknown"),
                            )
                        except Exception:
                            pass
                    st.toast(f'「{indicator_data.get("koumoku_name_full")}」を追加しました。')
                    st.rerun()
                else:
                    st.toast("この指標は既に追加されています。")

        with action_col2:
            base_url = (
                "https://app.powerbi.com/groups/f57d1ec6-4658-47f7-9a93-08811e43127f/reports/1accacdd-98d0-4d03-9b25-48f4c9673ff4/02fa5822008e814cf7f2?experience=power-bi"
            )
            indicator_code = indicator_data.get("koumoku_code", "")
            cleaned_indicator_code = indicator_code.lstrip("#")
            power_bi_url = (
                f"{base_url}&filter=social_demographic_pref_basic_bi/cat3_code eq '{cleaned_indicator_code}'"
            )

            if st.button("↗ Power BI で開く", key=f"powerbi_{unique_key}", type="secondary", use_container_width=True):
                if LOGGING_ENABLED:
                    try:
                        log_event(
                            session_id=state.get_session_id(),
                            event_type="open_powerbi",
                            user_query=state.get_original_query(),
                            selected_indicator=indicator_data,
                            selected_perspective=(state.get_selected_perspective() or {}).get(
                                "perspective_title", ""
                            ),
                            selected_group=st.session_state.get("selected_group_title", ""),
                            final_indicators=state.get_selected_group_indicators(),
                            llm_model=getattr(llm_config, "current_model", "unknown"),
                        )
                    except Exception:
                        pass

                if JAVASCRIPT_ENABLED:
                    st_javascript(f"window.open('{power_bi_url}', '_blank')")
                else:
                    st.info(
                        f"Power BIのURLをクリップボードにコピーしました。ブラウザで開いてください: {power_bi_url}"
                    )
                    if PYPERCLIP_AVAILABLE:
                        try:
                            pyperclip.copy(power_bi_url)  # type: ignore
                        except Exception:
                            pass


def render_initial_stage(state: StateManager) -> None:
    st.markdown("### 統計指標検索アシスタント")
    st.markdown("どのような統計指標をお探しですか？分析したいテーマを下のチャット欄に入力してください。")
    st.markdown("例: 子育て環境を比較したい、地域の教育水準を知りたい、高齢化の現状を把握したい")


def render_perspective_selection_stage(state: StateManager) -> None:
    st.markdown("### 分析観点の選択")
    st.markdown(f'「{state.get_original_query()}」について、どのような観点で分析しますか？')
    st.markdown("以下の選択肢から選択ボタンを押してください。")

    perspectives = (state.get_analysis_plan() or {}).get("perspectives", [])
    for i, perspective in enumerate(perspectives):
        with st.container(border=True):
            col_title, col_button = st.columns([4, 1])
            with col_title:
                st.markdown(f"**{i+1}. {perspective['perspective_title']}**")
            with col_button:
                if st.button("選択", key=f"perspective_{i}", type="primary", use_container_width=True):
                    state.set_selected_perspective(perspective)
                    state.add_message_to_history("user", f"{i+1}番目の{perspective['perspective_title']}について詳しく知りたいです")
                    state.set_current_options(perspective.get("suggested_groups", []))
                    state.set_stage(STAGE_GROUP_SELECTION)
                    state.add_message_to_history(
                        "assistant",
                        f'「{perspective["perspective_title"]}」ですね。この観点に関連する指標グループを以下から選択してください。',
                    )
                    st.rerun()
            st.markdown(perspective["perspective_description"])


def render_group_selection_stage(state: StateManager, services: AnalysisService) -> None:
    perspective = state.get_selected_perspective() or {}
    st.markdown("### 📊 指標グループの選択")
    st.markdown(f'「{perspective.get("perspective_title", "")}」について、より具体的な指標グループを以下から選択してください。')

    if not state.get_current_options():
        st.error("指標グループデータが取得できませんでした。")
        return

    for i, group in enumerate(state.get_current_options()):
        with st.container(border=True):
            col_title, col_button = st.columns([4, 1])
            with col_title:
                st.markdown(f"**{i+1}. {group['group_title']}**")
            with col_button:
                if st.button("選択", key=f"group_{i}", type="primary", use_container_width=True):
                    selected_group_title = group["group_title"]
                    state.add_message_to_history("user", f'「{selected_group_title}」グループの詳細が知りたい')

                    with st.spinner(f'「{selected_group_title}」グループの指標を検索・集計中...'):
                        search_results = services.generate_indicator_groups_for_perspective(selected_group_title)
                        if not search_results:
                            st.error("関連する指標グループが見つかりませんでした。")
                            st.stop()

                        # 代表グループの先頭を選ぶ（従来挙動を踏襲）
                        from retriever import retriever

                        res = retriever.hybrid_search(selected_group_title, top_k=5)
                        if not res:
                            st.error("関連する指標グループが見つかりませんでした。")
                            st.stop()
                        selected_koumoku_code = res[0]["koumoku_code"]

                        if retriever.df is not None:
                            group_indicators_df = retriever.df[
                                retriever.df["koumoku_code"].astype(str).str.startswith(str(selected_koumoku_code))
                            ].copy()
                            if not group_indicators_df.empty:
                                state.set_selected_group(
                                    str(selected_koumoku_code), group_indicators_df.to_dict("records"), selected_group_title
                                )
                                state.set_stage(STAGE_FINAL)
                                representative_name = group_indicators_df.iloc[0]["koumoku_name_full"]
                                state.add_message_to_history(
                                    "assistant", f'承知いたしました。「{representative_name}」に関連する指標を表示します。'
                                )
                            else:
                                st.error(f"指標グループ({selected_koumoku_code})に属する指標が見つかりませんでした。")
                        else:
                            st.error("データベースが読み込まれていません。")
                    st.rerun()
            st.markdown(group["group_description"])


@st.cache_data(show_spinner=False)
def _cached_group_summary_text(original_query: str, indicators: List[Dict[str, Any]]) -> str:
    # ダミー: 呼び出し側で streaming 済みテキストを保存・再利用するためのキャッシュキー用
    return "".join([original_query] + [i.get("koumoku_name_full", "") for i in indicators])


def render_final_stage(state: StateManager, services: AnalysisService) -> None:
    st.markdown("### 📊 指標グループ詳細")
    indicators = state.get_selected_group_indicators()
    if indicators:
        representative = indicators[0] if indicators else None
        if representative:
            st.markdown(f"**グループ**: {representative['koumoku_name_full']}関連指標")
            st.markdown(
                f"**分野**: {representative['bunya_name']} > {representative['chuubunrui_name']} > {representative['shoubunrui_name']}"
            )
        st.markdown(f"**該当指標数**: {len(indicators)}件")

        if state.get_original_query():
            if not st.session_state.get("summary_generated", False):
                with st.spinner("グループ要約を生成中..."):
                    stream_gen = services.stream_group_summary(indicators, state.get_original_query())
                    result_text = st.write_stream(stream_gen)
                    if result_text:
                        state.set_group_summary_text(result_text)
                        _ = _cached_group_summary_text(state.get_original_query(), indicators)
                # ストリーミング直後は同一実行内で再表示しない（重複防止）
            else:
                saved = state.get_group_summary_text()
                if saved:
                    st.divider()
                    st.markdown(saved)
                    st.divider()

        total = len(indicators)
        # 代表指標を除外せず、全件をそのまま表示する
        st.markdown(f"### 📊 関連指標一覧（{total}件）")
        for i, indicator_data in enumerate(indicators):
            display_indicator_card(state, indicator_data, "group", i)
    else:
        st.error("指標グループデータが見つかりません。")

    st.markdown("---")
    st.markdown("他にお探しの指標はありますか？")
    if st.button("🔄 新しい検索を開始", key="new_search", type="primary"):
        state.reset_session_state()
        st.rerun()

