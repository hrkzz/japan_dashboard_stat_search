from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from retriever import retriever
from llm_config import llm_config


class LLMService:
    """LLM 呼び出しを担う薄いサービス。UI に依存しない。"""

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        return llm_config.generate_response(messages, temperature=temperature)

    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.3):
        return llm_config.generate_response_stream(messages, temperature=temperature)


class AnalysisService:
    """アプリの中核ビジネスロジックを集約。UI から独立。"""

    def __init__(self, llm: Optional[LLMService] = None) -> None:
        self.llm = llm or LLMService()

    # --- Retrieval 補助 ---
    def get_available_indicators_for_query(self, query: str) -> str:
        try:
            if retriever.df is None:
                retriever.load_vector_database()

            logger.info(f"🔍 指標例取得開始: '{query}'")
            search_results = retriever.hybrid_search(query, top_k=40)

            bunya_groups: Dict[str, List[str]] = {}
            for result in search_results:
                bunya = result["bunya_name"]
                bunya_groups.setdefault(bunya, []).append(result["koumoku_name_full"])

            for bunya in list(bunya_groups.keys()):
                bunya_indicators = (
                    retriever.df[retriever.df["bunya_name"] == bunya]["koumoku_name_full"].tolist()
                )
                existing = set(bunya_groups[bunya])
                additional = [ind for ind in bunya_indicators if ind not in existing][:10]
                bunya_groups[bunya].extend(additional)

            indicator_examples: List[str] = []
            for bunya, indicators in bunya_groups.items():
                indicator_examples.append(
                    f"【{bunya}】({len(indicators)}件利用可能): {', '.join(indicators[:15])}"
                )

            return "\n".join(indicator_examples)
        except Exception as e:
            return f"指標リスト取得エラー: {str(e)}"

    # --- LLM 生成系 ---
    def generate_analysis_plan(self, query: str) -> Optional[Dict[str, Any]]:
        logger.info(f"🤖 分析計画の生成開始: '{query}'")
        available_indicators = self.get_available_indicators_for_query(query)

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
        user_prompt = (
            f"以下の質問について、詳細な分析計画をJSON形式で提案してください：\n\n{query}"
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = self.llm.generate(messages, temperature=0.2)

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result: Dict[str, Any] = json.loads(json_match.group())
                if "analysis_plan" in result and "perspectives" in result["analysis_plan"]:
                    result["analysis_plan"]["perspectives"] = result["analysis_plan"][
                        "perspectives"
                    ][:10]
                logger.info("✅ 分析計画の生成に成功しました。")
                return result
            else:
                logger.error(f"❌ 分析計画のJSON生成に失敗: {response[:500]}...")
                return None
        except Exception as e:
            logger.error(f"❌ 分析計画の生成エラー: {str(e)}")
            return None

    def stream_analysis_plan_raw(self, query: str):
        available_indicators = self.get_available_indicators_for_query(query)
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.stream(messages, temperature=0.2)

    def generate_group_summary(self, group_indicators: List[Dict[str, Any]], user_query: str) -> str:
        logger.info(
            f"🤖 グループ要約生成開始: '{user_query}' for {len(group_indicators)} indicators"
        )

        indicator_names = [i.get("koumoku_name_full", "") for i in group_indicators]
        representative = group_indicators[0] if group_indicators else {}
        bunya_info = (
            f"{representative.get('bunya_name', '')} > {representative.get('chuubunrui_name', '')} > {representative.get('shoubunrui_name', '')}"
        )

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

        user_prompt = (
            f"上記の指標グループについて、ユーザーのクエリ「{user_query}」に対する要約説明文を生成してください。"
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = self.llm.generate(messages, temperature=0.3)
            logger.info("✅ グループ要約生成成功")
            return response.strip()
        except Exception as e:
            logger.error(f"❌ グループ要約生成エラー: {str(e)}")
            return "このグループの指標は、ユーザーの関心事に関連する重要な統計データを含んでいます。"

    def stream_group_summary(self, group_indicators: List[Dict[str, Any]], user_query: str):
        indicator_names = [i.get("koumoku_name_full", "") for i in group_indicators]
        representative = group_indicators[0] if group_indicators else {}
        bunya_info = (
            f"{representative.get('bunya_name', '')} > {representative.get('chuubunrui_name', '')} > {representative.get('shoubunrui_name', '')}"
        )
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
        user_prompt = (
            f"上記の指標グループについて、ユーザーのクエリ「{user_query}」に対する要約説明文を生成してください。"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.stream(messages, temperature=0.3)

    def generate_indicator_explanations(
        self, user_query: str, indicators_list: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        logger.info(
            f"🤖 指標説明文生成開始: '{user_query}' for {len(indicators_list)} indicators"
        )

        indicator_names = [i.get("koumoku_name_full", "") for i in indicators_list]
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

        user_prompt = (
            f"上記の統計指標について、ユーザーのクエリ「{user_query}」との関連性を説明してください。"
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = self.llm.generate(messages, temperature=0.3)
            logger.info(f"🔍 指標説明文LLM応答の最初の500文字: {response[:500]}")

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                logger.info(f"✅ 指標説明文生成成功: {len(parsed_json)}件")
                return parsed_json
            else:
                logger.error(f"❌ 有効なJSONが生成されませんでした: {response[:500]}")
                return {}
        except Exception as e:
            logger.error(f"❌ 指標説明文生成エラー: {str(e)}")
            return {}

    def generate_indicator_groups_for_perspective(
        self, perspective_title: str
    ) -> Optional[Dict[str, Any]]:
        logger.info(f"🤖 指標グループ生成開始: '{perspective_title}'")
        try:
            if retriever.df is None:
                retriever.load_vector_database()

            logger.info("📍 ステップA: hybrid_searchで関連指標を取得")
            search_results = retriever.hybrid_search(perspective_title, top_k=80)
            if not search_results:
                logger.error("❌ hybrid_searchで結果が取得できませんでした")
                return None

            related_indicator_names = [r["koumoku_name_full"] for r in search_results]
            logger.info(f"📊 hybrid_searchで取得した関連指標数: {len(related_indicator_names)}件")

            df_filtered = retriever.df[retriever.df["koumoku_name_full"].isin(related_indicator_names)]
            logger.info(f"📋 一次フィルタリング済みDataFrame: {len(df_filtered)}行")
            if df_filtered.empty:
                logger.warning("⚠️ 一次フィルタリング済みDataFrameが空です")
                return None

            logger.info("📍 ステップB: ユニークなgroup_codeを抽出")
            group_codes = df_filtered["group_code"].dropna().unique()
            logger.info(f"🔍 抽出されたグループコード: {group_codes.tolist()}")
            logger.info(f"🔍 グループコード数: {len(group_codes)}件")

            group_indicators: List[Dict[str, Any]] = []
            for group_code in sorted(group_codes):
                group_code_str = str(group_code)

                representative = retriever.df[
                    retriever.df["koumoku_code"].astype(str) == group_code_str
                ]
                if not representative.empty:
                    row = representative.iloc(0)[0] if hasattr(representative, "iloc") else representative.iloc[0]
                    group_indicators.append(
                        {
                            "group_code": group_code_str,
                            "title": row["koumoku_name_full"],
                            "description": f"「{row['koumoku_name_full']}」グループに含まれる全ての関連指標",
                        }
                    )
                else:
                    fallback_indicators = df_filtered[
                        df_filtered["group_code"].astype(str) == group_code_str
                    ]
                    if not fallback_indicators.empty:
                        row = fallback_indicators.iloc[0]
                        group_indicators.append(
                            {
                                "group_code": group_code_str,
                                "title": f"{row['koumoku_name_full']}関連グループ",
                                "description": f"「{row['koumoku_name_full']}」関連の指標グループ",
                            }
                        )

            group_indicators = group_indicators[:15]
            logger.info(f"✅ {len(group_indicators)}個の指標グループを生成")
            return {"groups": group_indicators}
        except Exception as e:
            logger.error(f"❌ 指標グループ生成エラー: {str(e)}")
            return None

    # 速度比較スクリプト互換のユーティリティ（必要に応じて使用）
    def generate_ai_analysis(self, query: str) -> Optional[Dict[str, Any]]:
        """速度テスト互換の簡易 API。

        返却形式は { 'analysis_perspectives': [ { 'recommended_indicators': [...] }, ... ] }
        に整形する。
        """
        plan = self.generate_analysis_plan(query)
        if not plan or "analysis_plan" not in plan:
            return None

        perspectives = plan["analysis_plan"].get("perspectives", [])
        normalized: List[Dict[str, Any]] = []
        for p in perspectives:
            groups = p.get("suggested_groups", [])
            normalized.append({"recommended_indicators": groups})
        return {"analysis_perspectives": normalized}

