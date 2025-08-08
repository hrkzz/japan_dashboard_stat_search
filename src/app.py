import streamlit as st
import os

from retriever import retriever
from llm_config import llm_config
from state_manager import (
    StateManager,
    STAGE_INITIAL,
    STAGE_PERSPECTIVE_SELECTION,
    STAGE_GROUP_SELECTION,
    STAGE_FINAL,
)
from services import AnalysisService
from ui_components import (
    render_initial_stage,
    render_perspective_selection_stage,
    render_group_selection_stage,
    render_final_stage,
)


def main() -> None:
    """Streamlit UI entry point. Keeps UI-only responsibilities."""
    st.set_page_config(page_title="統計指標検索アシスタント", page_icon="", layout="wide")

    # 外部 CSS 読み込み
    try:
        css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    state = StateManager()
    services = AnalysisService()
    state.initialize_session_state()

    # データベースの初期化
    with st.spinner("📚 統計データベースを初期化中..."):
        if not retriever.load_vector_database():
            st.error("❌ データベースの読み込みに失敗しました")
            st.stop()

    # サイドバー
    with st.sidebar:
        available_models = llm_config.get_available_models()
        if available_models:
            model_options = list(available_models.keys())
            current_model_display = next(
                (k for k, v in available_models.items() if v == llm_config.current_model), None
            )
            selected_model_display = st.selectbox(
                "AIモデル選択",
                model_options,
                index=model_options.index(current_model_display) if current_model_display else 0,
            )
            selected_model = available_models[selected_model_display]
            if selected_model != llm_config.current_model:
                llm_config.set_model(selected_model)

        st.header("保存リスト")
        saved = state.get_saved_indicators()
        if saved:
            for i, indicator_data in enumerate(saved):
                st.markdown(f"**{i+1}. {indicator_data.get('koumoku_name_full', '')}**")
                btn_col1, btn_col2 = st.columns(2, gap="small")
                with btn_col1:
                    base_url = "https://app.powerbi.com/groups/f57d1ec6-4658-47f7-9a93-08811e43127f/reports/1accacdd-98d0-4d03-9b25-48f4c9673ff4/02fa5822008e814cf7f2?experience=power-bi"
                    indicator_code = indicator_data.get("koumoku_code", "")
                    cleaned_indicator_code = indicator_code.lstrip("#")
                    power_bi_url = f"{base_url}&filter=social_demographic_pref_basic_bi/cat3_code eq '{cleaned_indicator_code}'"
                    st.link_button("↗ リンク", power_bi_url, use_container_width=True)
                with btn_col2:
                    if st.button("🗑️ 削除", key=f"delete_{i}", use_container_width=True):
                        state.remove_saved_indicator_at(i)
                        st.rerun()
                st.markdown("---")
        else:
            st.info("まだ指標が採用されていません。")
            st.markdown("気になる指標の「＋ 保存リストへ」ボタンをクリックして、このリストに保存できます。")

    # メインコンテンツ
    stage = state.get_stage()
    if stage == STAGE_INITIAL:
        render_initial_stage(state)
        user_input = st.chat_input("分析したいテーマを入力してください（例：子育て環境を比較したい）")
        if user_input:
            state.add_message_to_history("user", user_input)
            with st.spinner("分析計画を調査中..."):
                plan_result = services.generate_analysis_plan(user_input)
                if plan_result and "analysis_plan" in plan_result:
                    state.set_analysis_plan(plan_result["analysis_plan"])
                    state.set_original_query(user_input)
                    state.set_stage(STAGE_PERSPECTIVE_SELECTION)
                    state.add_message_to_history(
                        "assistant", f"承知いたしました。「{user_input}」についてですね。どのような観点で分析しますか？"
                    )
                else:
                    state.add_message_to_history(
                        "assistant", "申し訳ございません。分析計画の生成に失敗しました。もう一度お試しください。"
                    )
            st.rerun()
    elif stage == STAGE_PERSPECTIVE_SELECTION:
        render_perspective_selection_stage(state)
    elif stage == STAGE_GROUP_SELECTION:
        render_group_selection_stage(state, services)
    elif stage == STAGE_FINAL:
        render_final_stage(state, services)


if __name__ == "__main__":
    main()

