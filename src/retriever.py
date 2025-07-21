import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import json
import os
from typing import List, Dict, Tuple, Optional

from encoder import EmbeddingConfig
from loguru import logger

class HybridRetriever:
    """ハイブリッド検索（ベクトル検索 + キーワード検索）とリランキングを行うクラス"""
    
    def __init__(self):
        self.df = None
        self.faiss_index = None
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.metadata = None
    
    def load_vector_database(self, vector_db_dir: str = '../vector_db') -> bool:
        """ベクトルデータベースとメタデータを読み込む"""
        try:
            # 既に読み込み済みの場合はスキップ
            if (self.df is not None and hasattr(self, 'index') and self.index is not None 
                and hasattr(self, 'bm25_index') and self.bm25_index is not None):
                logger.info("🎯 データベースは既に読み込み済み")
                return True
                
            logger.info(f"🔧 データベース読み込み開始: {vector_db_dir}")
            
            # メタデータファイルのパス
            metadata_path = os.path.join(vector_db_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                logger.error(f"❌ メタデータファイルが見つかりません: {metadata_path}")
                return False
            
            # メタデータを読み込み
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ メタデータ読み込み完了")
            
            # Parquetファイルを読み込み
            parquet_path = os.path.join(vector_db_dir, 'processed_data.parquet')
            if not os.path.exists(parquet_path):
                logger.error(f"❌ Parquetファイルが見つかりません: {parquet_path}")
                return False
            
            self.df = pd.read_parquet(parquet_path)
            logger.info(f"✅ データフレーム読み込み完了: {len(self.df)}行")
            
            # FAISSインデックスを読み込み
            index_path = os.path.join(vector_db_dir, 'faiss_index.bin')
            if not os.path.exists(index_path):
                logger.error(f"❌ FAISSインデックスが見つかりません: {index_path}")
                return False
            
            self.index = faiss.read_index(index_path)
            logger.info(f"✅ FAISSインデックス読み込み完了: {self.index.ntotal}件")
            
            # エンベディング設定を初期化
            self.embedding_config = EmbeddingConfig()
            logger.info(f"✅ エンベディング設定初期化完了")
            
            # キーワード検索用のインデックスを構築
            self._build_keyword_indices()
            logger.info(f"✅ キーワードインデックス構築完了")
            
            logger.info(f"🎯 データベース読み込み全体完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ データベース読み込みエラー: {str(e)}")
            import traceback
            logger.error(f"❌ スタックトレース: {traceback.format_exc()}")
            return False
    
    def _build_keyword_indices(self):
        """BM25とTF-IDFインデックスを構築"""
        try:
            # 検索対象テキストの作成
            search_texts = []
            for _, row in self.df.iterrows():
                text = f"{row['koumoku_name_full']} {row['bunya_name']} {row['chuubunrui_name']} {row['shoubunrui_name']} {row['definition']} {row['stat_name']}"
                search_texts.append(text)
            
            # BM25インデックス
            tokenized_texts = [text.split() for text in search_texts]
            self.bm25 = BM25Okapi(tokenized_texts)
            
            # TF-IDFインデックス
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=None  # 日本語対応のため
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(search_texts)
            
        except Exception as e:
            st.error(f"キーワードインデックス構築エラー: {str(e)}")
    
    def vector_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """ベクトル検索を実行"""
        try:
            if self.index is None or self.embedding_config is None:
                logger.error("❌ ベクトル検索: インデックスまたはエンベディング設定が未初期化")
                return []
            
            # クエリのエンベディングを取得
            query_embedding = self.embedding_config.get_embeddings([query])
            
            if query_embedding.size == 0:
                logger.error("❌ ベクトル検索: クエリエンベディングが取得できませんでした")
                return []
            
            # FAISS検索
            distances, indices = self.index.search(query_embedding, top_k)
            
            # 結果を整形
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # 有効なインデックス
                    similarity = 1.0 / (1.0 + distance)  # 距離を類似度に変換
                    results.append((idx, similarity))
            
            logger.info(f"🔍 ベクトル検索完了: {len(results)}件 (要求:{top_k}件)")
            return results
            
        except Exception as e:
            logger.error(f"❌ ベクトル検索エラー: {str(e)}")
            st.error(f"ベクトル検索エラー: {str(e)}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25キーワード検索を実行"""
        try:
            if self.bm25 is None:
                logger.error("❌ BM25検索: BM25インデックスが未初期化")
                return []
                
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # スコア順にソート
            indexed_scores = [(i, score) for i, score in enumerate(bm25_scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            results = indexed_scores[:top_k]
            logger.info(f"🔍 BM25検索完了: {len(results)}件 (要求:{top_k}件)")
            return results
            
        except Exception as e:
            logger.error(f"❌ キーワード検索エラー: {str(e)}")
            st.error(f"キーワード検索エラー: {str(e)}")
            return []
    
    def tfidf_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """TF-IDF検索を実行"""
        try:
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                logger.error("❌ TF-IDF検索: TF-IDFインデックスが未初期化")
                return []
                
            query_vector = self.tfidf_vectorizer.transform([query])
            cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            indexed_scores = [(i, score) for i, score in enumerate(cosine_scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            results = indexed_scores[:top_k]
            logger.info(f"🔍 TF-IDF検索完了: {len(results)}件 (要求:{top_k}件)")
            return results
            
        except Exception as e:
            logger.error(f"❌ TF-IDF検索エラー: {str(e)}")
            st.error(f"TF-IDF検索エラー: {str(e)}")
            return []
    
    def rerank_results(self, query: str, candidate_indices: List[int], top_k: int = 50) -> List[int]:
        """シンプルなリランキング（クエリとの類似度ベース）"""
        try:
            query_lower = query.lower()
            scored_candidates = []
            
            for idx in candidate_indices:
                row = self.df.iloc[idx]
                
                # 各フィールドとのマッチング度を計算
                score = 0
                text_fields = [
                    row['koumoku_name_full'],
                    row['bunya_name'],
                    row['chuubunrui_name'],
                    row['shoubunrui_name'],
                    row['definition'],
                    row['stat_name']
                ]
                
                for field in text_fields:
                    if pd.isna(field):
                        continue
                    field_lower = str(field).lower()
                    
                    # 完全一致ボーナス
                    if query_lower in field_lower:
                        score += 2
                    
                    # 部分一致
                    query_words = query_lower.split()
                    field_words = field_lower.split()
                    matches = len(set(query_words) & set(field_words))
                    score += matches
                
                scored_candidates.append((idx, score))
            
            # スコア順にソートして上位を返す
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in scored_candidates[:top_k]]
            
        except Exception as e:
            st.error(f"リランキングエラー: {str(e)}")
            return candidate_indices[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 50, vector_weight: float = 0.6) -> List[Dict]:
        """ハイブリッド検索を実行"""
        try:
            logger.info(f"🔍 クエリ: '{query}' (top_k={top_k})")
            
            # 各検索手法を実行
            # 各検索手法で結果を取得
            vector_results = self.vector_search(query, top_k * 2)
            bm25_results = self.keyword_search(query, top_k * 2)
            tfidf_results = self.tfidf_search(query, top_k * 2)
            
            logger.info(f"📊 検索結果数: ベクトル={len(vector_results)}, BM25={len(bm25_results)}, TF-IDF={len(tfidf_results)}")
            
            # スコアを正規化してマージ
            all_candidates = {}
            keyword_weight = (1 - vector_weight) / 2
            
            # ベクトル検索結果
            for idx, score in vector_results:
                all_candidates[idx] = all_candidates.get(idx, 0) + score * vector_weight
            
            # BM25結果
            max_bm25 = max([score for _, score in bm25_results], default=1)
            for idx, score in bm25_results:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                all_candidates[idx] = all_candidates.get(idx, 0) + normalized_score * keyword_weight
            
            # TF-IDF結果
            max_tfidf = max([score for _, score in tfidf_results], default=1)
            for idx, score in tfidf_results:
                normalized_score = score / max_tfidf if max_tfidf > 0 else 0
                all_candidates[idx] = all_candidates.get(idx, 0) + normalized_score * keyword_weight
            
            # スコア順にソート
            sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
            candidate_indices = [idx for idx, _ in sorted_candidates[:top_k * 2]]
            
            logger.info(f"🔄 リランキング前の候補数: {len(candidate_indices)}")
            
            # リランキング設定の問題を解決 - より多くの結果を返すように
            final_top_k = min(top_k, 80)  # 40から80に増加
            reranked_indices = self.rerank_results(query, candidate_indices, final_top_k)
            
            logger.info(f"✅ リランキング後の結果数: {len(reranked_indices)} (最大{final_top_k})")
            
            # 結果を整形
            results = []
            bunya_counts = {}
            
            for idx in reranked_indices:
                item = self.df.iloc[idx]
                bunya = item['bunya_name']
                bunya_counts[bunya] = bunya_counts.get(bunya, 0) + 1
                
                results.append({
                    'koumoku_name': item.get('koumoku_name', item['koumoku_name_full']),
                    'koumoku_name_full': item['koumoku_name_full'],
                    'bunya_name': item['bunya_name'],
                    'chuubunrui_name': item['chuubunrui_name'],
                    'shoubunrui_name': item['shoubunrui_name'],
                    'score': all_candidates.get(idx, 0)
                })
            
            logger.info(f"📈 最終結果の分野分布: {dict(bunya_counts)}")
            
            # 詳細結果をログ出力
            logger.info(f"📋 検索結果詳細:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i:2d}. {result['koumoku_name_full']} ({result['bunya_name']})")
            
            logger.info(f"🎯 検索完了: {len(results)}件の指標を返却")
            
            return results
            
        except Exception as e:
            st.error(f"ハイブリッド検索エラー: {str(e)}")
            return []

# グローバルインスタンス
retriever = HybridRetriever() 