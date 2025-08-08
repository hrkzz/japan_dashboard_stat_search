from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from loguru import logger


STAGE_INITIAL = "initial"
STAGE_PERSPECTIVE_SELECTION = "perspective_selection"
STAGE_GROUP_SELECTION = "group_selection"
STAGE_FINAL = "final"


class StateManager:
    """Streamlit の session_state 操作をカプセル化する。

    UI 以外の層が `st.session_state` に直接依存しないようにするための薄いラッパー。
    """

    def __init__(self) -> None:
        self._state = st.session_state

    def initialize_session_state(self) -> None:
        if "session_id" not in self._state:
            self._state.session_id = str(uuid.uuid4())
            logger.info(f"🆔 新しいセッションID生成: {self._state.session_id[:8]}...")

        self._state.setdefault("stage", STAGE_INITIAL)
        self._state.setdefault("chat_history", [])
        self._state.setdefault("current_options", [])
        self._state.setdefault("selected_perspective", None)
        self._state.setdefault("original_query", "")
        self._state.setdefault("available_indicators", "")
        self._state.setdefault("selected_group_code", None)
        self._state.setdefault("selected_group_indicators", [])
        self._state.setdefault("analysis_plan", None)
        self._state.setdefault("saved_indicators", [])
        self._state.setdefault("summary_generated", False)
        self._state.setdefault("generated_summary_text", "")
        self._state.setdefault("selected_group_title", "")

    def reset_session_state(self) -> None:
        logger.info("🔄 セッション状態をリセット")
        for key in [
            "stage",
            "current_options",
            "selected_perspective",
            "original_query",
            "available_indicators",
            "selected_group_code",
            "selected_group_indicators",
            "analysis_plan",
            "session_id",
            "summary_generated",
            "generated_summary_text",
            "selected_group_title",
        ]:
            if key in self._state:
                del self._state[key]

        self._state.stage = STAGE_INITIAL
        self._state.current_options = []
        self._state.session_id = str(uuid.uuid4())
        logger.info(f"🆔 新しいセッションID生成: {self._state.session_id[:8]}...")

    # --- History ---
    def add_message_to_history(self, role: str, content: str) -> None:
        self._state.chat_history.append(
            {"role": role, "content": content, "timestamp": time.time()}
        )

    # --- Getters/Setters ---
    def get_stage(self) -> str:
        return self._state.stage

    def set_stage(self, stage: str) -> None:
        self._state.stage = stage

    def get_session_id(self) -> str:
        return self._state.session_id

    def set_analysis_plan(self, plan: Dict[str, Any]) -> None:
        self._state.analysis_plan = plan

    def get_analysis_plan(self) -> Optional[Dict[str, Any]]:
        return self._state.analysis_plan

    def set_original_query(self, query: str) -> None:
        self._state.original_query = query

    def get_original_query(self) -> str:
        return self._state.original_query

    def set_selected_perspective(self, perspective: Dict[str, Any]) -> None:
        self._state.selected_perspective = perspective

    def get_selected_perspective(self) -> Optional[Dict[str, Any]]:
        return self._state.selected_perspective

    def set_current_options(self, options: List[Dict[str, Any]]) -> None:
        self._state.current_options = options

    def get_current_options(self) -> List[Dict[str, Any]]:
        return self._state.current_options

    def set_selected_group(self, code: str, indicators: List[Dict[str, Any]], title: str) -> None:
        self._state.selected_group_code = code
        self._state.selected_group_indicators = indicators
        self._state.selected_group_title = title
        self._state.summary_generated = False

    def get_selected_group_indicators(self) -> List[Dict[str, Any]]:
        return self._state.selected_group_indicators

    def get_selected_group_code(self) -> Optional[str]:
        return self._state.get("selected_group_code")

    def set_available_indicators_text(self, text: str) -> None:
        self._state.available_indicators = text

    def get_available_indicators_text(self) -> str:
        return self._state.available_indicators

    def add_saved_indicator(self, indicator: Dict[str, Any]) -> None:
        self._state.saved_indicators.append(indicator)

    def remove_saved_indicator_at(self, index: int) -> None:
        self._state.saved_indicators.pop(index)

    def get_saved_indicators(self) -> List[Dict[str, Any]]:
        return self._state.saved_indicators

    def set_group_summary_text(self, text: str) -> None:
        self._state.generated_summary_text = text
        self._state.summary_generated = True

    def get_group_summary_text(self) -> str:
        return self._state.get("generated_summary_text", "")

