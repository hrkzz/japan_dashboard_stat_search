#!/usr/bin/env python3
"""
OpenAI GPT-4o-mini vs Google Gemini 2.0 Flash 速度比較テスト
"""

import sys
import os
# プロジェクトルートをパスに追加（src をパッケージとして解決するため）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from loguru import logger
from services import AnalysisService
from llm_config import llm_config

# ログ設定
logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level} | {message}')

def test_model_speed(model_name: str, model_display_name: str, test_query: str) -> dict:
    """指定されたモデルでの速度テスト"""
    logger.info(f"🚀 {model_display_name} テスト開始")
    
    # モデル切り替え
    llm_config.set_model(model_name)
    
    # 3回テストして平均を取る
    times = []
    results = []
    
    for i in range(3):
        logger.info(f"  テスト {i+1}/3 実行中...")
        start_time = time.time()
        
        try:
            service = AnalysisService()
            result = service.generate_ai_analysis(test_query)
            end_time = time.time()
            
            if result and 'analysis_perspectives' in result:
                test_time = end_time - start_time
                times.append(test_time)
                total_indicators = sum(len(p['recommended_indicators']) for p in result['analysis_perspectives'])
                results.append(total_indicators)
                logger.info(f"    テスト {i+1}: {test_time:.2f}秒, 指標数: {total_indicators}件")
            else:
                logger.error(f"    テスト {i+1}: 失敗")
                times.append(None)
                results.append(0)
        except Exception as e:
            logger.error(f"    テスト {i+1}: エラー - {str(e)}")
            times.append(None)
            results.append(0)
    
    # 結果を計算
    valid_times = [t for t in times if t is not None]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else None
    avg_indicators = sum(results) / len(results) if results else 0
    
    return {
        'model': model_display_name,
        'times': times,
        'avg_time': avg_time,
        'avg_indicators': avg_indicators,
        'success_rate': len(valid_times) / len(times) * 100
    }

def main():
    logger.info("=" * 60)
    logger.info("🏁 OpenAI vs Gemini 速度比較テスト開始")
    logger.info("=" * 60)
    
    # テストクエリ
    test_queries = [
        "子育て環境を比較したい",
        "地域の教育水準を知りたい",
        "高齢化の現状を把握したい"
    ]
    
    # 利用可能モデル取得
    available_models = llm_config.get_available_models()
    logger.info(f"利用可能モデル: {list(available_models.keys())}")
    
    if len(available_models) < 2:
        logger.error("比較するには2つ以上のモデルが必要です")
        return
    
    results = []
    
    # 各クエリでテスト
    for query_idx, query in enumerate(test_queries, 1):
        logger.info(f"\n📊 クエリ {query_idx}/3: '{query}'")
        logger.info("-" * 40)
        
        query_results = []
        
        # 各モデルでテスト
        for display_name, model_name in available_models.items():
            result = test_model_speed(model_name, display_name, query)
            query_results.append(result)
            
            if result['avg_time']:
                logger.info(f"✅ {display_name}: 平均 {result['avg_time']:.2f}秒")
            else:
                logger.error(f"❌ {display_name}: テスト失敗")
        
        results.append({
            'query': query,
            'results': query_results
        })
    
    # 総合結果
    logger.info("\n" + "=" * 60)
    logger.info("📈 総合結果")
    logger.info("=" * 60)
    
    # モデル別の平均計算
    model_stats = {}
    for model_display, _ in available_models.items():
        times = []
        indicators = []
        success_count = 0
        
        for query_result in results:
            for model_result in query_result['results']:
                if model_result['model'] == model_display:
                    if model_result['avg_time']:
                        times.append(model_result['avg_time'])
                        indicators.append(model_result['avg_indicators'])
                        success_count += 1
        
        if times:
            model_stats[model_display] = {
                'avg_time': sum(times) / len(times),
                'avg_indicators': sum(indicators) / len(indicators),
                'success_rate': success_count / len(test_queries) * 100
            }
    
    # 結果表示
    for model, stats in model_stats.items():
        logger.info(f"\n🤖 {model}:")
        logger.info(f"  ⏱️  平均処理時間: {stats['avg_time']:.2f}秒")
        logger.info(f"  📊 平均指標数: {stats['avg_indicators']:.1f}件")
        logger.info(f"  ✅ 成功率: {stats['success_rate']:.0f}%")
    
    # 勝者判定
    if len(model_stats) >= 2:
        fastest = min(model_stats.items(), key=lambda x: x[1]['avg_time'])
        logger.info(f"\n🏆 最高速: {fastest[0]} ({fastest[1]['avg_time']:.2f}秒)")
        
        # 速度差計算
        times_list = [(name, stats['avg_time']) for name, stats in model_stats.items()]
        times_list.sort(key=lambda x: x[1])
        
        if len(times_list) >= 2:
            speed_improvement = ((times_list[1][1] - times_list[0][1]) / times_list[1][1] * 100)
            logger.info(f"🚀 速度差: {speed_improvement:.1f}%高速化")

if __name__ == "__main__":
    main() 