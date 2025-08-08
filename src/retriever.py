import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import streamlit as st
import json
import os
import requests
import zipfile
from io import BytesIO
from typing import List, Dict, Tuple
from encoder import EmbeddingConfig
from loguru import logger
from config import config
import time

@st.cache_data(ttl=3600, show_spinner=False) # 1時間キャッシュする
def load_db_from_github(zip_url: str):
    """
    GitHub Releasesからzipをダウンロードし、中のファイルをメモリにロードする。
    """
    logger.info(f"⬇️ GitHub Releasesからデータベースをダウンロード開始: {zip_url}")
    try:
        response = requests.get(zip_url)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # zipファイル内のファイル名を特定
            parquet_filename = next(name for name in z.namelist() if name.endswith('processed_data.parquet'))
            faiss_filename = next(name for name in z.namelist() if name.endswith('faiss_index.bin'))

            # ファイルをメモリ上で読み込む
            with z.open(parquet_filename) as pf:
                df = pd.read_parquet(pf)
            
            # group_code列を追加（koumoku_codeの先頭5文字）
            if 'koumoku_code' in df.columns:
                # koumoku_codeを文字列型に変換してから先頭5文字を取得
                df['group_code'] = df['koumoku_code'].astype(str).str[:5]
                logger.info(f"✅ group_code列を追加しました（{df['group_code'].nunique()}個のグループ）")
            else:
                logger.warning("⚠️ koumoku_code列が見つかりません")
            
            with z.open(faiss_filename) as ff:
                # faissはファイルパスを要求するため、一時ファイルに書き出す
                temp_faiss_path = "temp_faiss_index.bin"
                with open(temp_faiss_path, "wb") as f_out:
                    f_out.write(ff.read())
                faiss_index = faiss.read_index(temp_faiss_path)
                os.remove(temp_faiss_path) # 一時ファイルを削除

        logger.info("✅ データベースのダウンロードと読み込みが完了")
        return df, faiss_index

    except Exception as e:
        logger.error(f"❌ GitHubからのDBロードエラー: {e}")
        return None, None

class HybridRetriever:
    """ハイブリッド検索（ベクトル検索 + キーワード検索）とリランキングを行うクラス"""
    
    def __init__(self):
        self.df = None
        self.faiss_index = None # 以前は `index` だったものを `faiss_index` に統一
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embedding_config = EmbeddingConfig()

    def load_vector_database(self) -> bool:
        """GitHub Releasesからベクトルデータベースを読み込む"""
        if self.df is not None and self.faiss_index is not None:
            return True

        # Config 管理の ZIP URL（環境変数で上書き可能）
        zip_url = config.get_vector_db_zip_url()
        
        # GitHubからDBをロード
        self.df, self.faiss_index = load_db_from_github(zip_url)

        if self.df is None or self.faiss_index is None:
            return False

        self._build_keyword_indices()
        logger.info("🎯 データベースの初期化が完了しました")
        return True
    
    def _build_keyword_indices(self):
        """BM25/TF-IDF インデックスをロード。なければ初回のみ作成して永続化。"""
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vector_db'))
            os.makedirs(base_dir, exist_ok=True)
            bm25_path = os.path.join(base_dir, 'bm25.joblib')
            tfidf_path = os.path.join(base_dir, 'tfidf.joblib')

            loaded_any = False
            if os.path.exists(bm25_path):
                self.bm25 = joblib.load(bm25_path)
                loaded_any = True
            if os.path.exists(tfidf_path):
                tfidf_bundle = joblib.load(tfidf_path)
                self.tfidf_vectorizer = tfidf_bundle.get('vectorizer')
                self.tfidf_matrix = tfidf_bundle.get('matrix')
                loaded_any = True

            if not loaded_any:
                logger.info("🔧 キーワードインデックスが未検出のため初回構築を行います…")
                # 検索用テキストを組み立て（以前のロジックを踏襲）
                search_texts = []
                for _, row in self.df.iterrows():
                    text = f"{row['koumoku_name_full']} {row['bunya_name']} {row['chuubunrui_name']} {row['shoubunrui_name']} {row['definition']} {row['stat_name']}"
                    search_texts.append(text)

                # BM25
                tokenized_texts = [text.split() for text in search_texts]
                self.bm25 = BM25Okapi(tokenized_texts)

                # TF-IDF
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words=None
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(search_texts)

                # 永続化
                try:
                    joblib.dump(self.bm25, bm25_path)
                    joblib.dump({
                        'vectorizer': self.tfidf_vectorizer,
                        'matrix': self.tfidf_matrix
                    }, tfidf_path)
                    logger.info("✅ BM25/TF-IDF インデックスを保存しました。次回以降の起動が高速化されます。")
                except Exception as persist_err:
                    logger.warning(f"⚠️ インデックス保存に失敗: {persist_err}")
        except Exception as e:
            logger.error(f"❌ キーワードインデックス読み込み/構築エラー: {str(e)}")
    
    def vector_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """ベクトル検索を実行"""
        start_time = time.time()
        if self.faiss_index is None or self.embedding_config is None:
            logger.error("❌ ベクトル検索: インデックスまたはエンベディング設定が未初期化")
            return []

        query_embedding = self.embedding_config.get_embeddings([query])

        if query_embedding.size == 0:
            logger.error("❌ ベクトル検索: クエリエンベディングが取得できませんでした")
            return []

        # FAISS検索
        distances, indices = self.faiss_index.search(query_embedding, top_k)
            
        # 結果を整形
        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # 有効なインデックス
                results.append((idx, score))
            
        logger.info(f"🔍 ベクトル検索完了: {len(results)}件 (要求:{top_k}件)")
        logger.debug(f"⏱ vector_search took {(time.time()-start_time)*1000:.1f} ms")
        return results
    
    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25キーワード検索を実行"""
        start_time = time.time()
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
            logger.debug(f"⏱ keyword_search took {(time.time()-start_time)*1000:.1f} ms")
            return results
            
        except Exception as e:
            logger.error(f"❌ キーワード検索エラー: {str(e)}")
            st.error(f"キーワード検索エラー: {str(e)}")
            return []
    
    def tfidf_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """TF-IDF検索を実行"""
        start_time = time.time()
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
            logger.debug(f"⏱ tfidf_search took {(time.time()-start_time)*1000:.1f} ms")
            return results
            
        except Exception as e:
            logger.error(f"❌ TF-IDF検索エラー: {str(e)}")
            st.error(f"TF-IDF検索エラー: {str(e)}")
            return []
    
    def rerank_results(self, query: str, candidate_indices: List[int], top_k: int = 50) -> List[int]:
        """シンプルなリランキング（NumPy/Pandas寄りに軽量化）。"""
        start_time = time.time()
        try:
            query_lower = query.lower()
            # ベクトル化（最低限の高速化）
            text_cols = ['koumoku_name_full', 'bunya_name', 'chuubunrui_name', 'shoubunrui_name', 'definition', 'stat_name']
            # 対象行だけ抽出
            sub_df = self.df.iloc[candidate_indices][text_cols].fillna("").astype(str)
            # 完全一致: 各列に query を含むか
            contain_mask = sub_df.applymap(lambda x: query_lower in x.lower())
            contain_score = contain_mask.sum(axis=1) * 2
            # 単語一致: 単純単語分割して集合積
            q_words = set(query_lower.split())
            def word_overlap_score(row: pd.Series) -> int:
                s = 0
                for v in row.values:
                    fw = set(str(v).lower().split())
                    s += len(q_words & fw)
                return s
            overlap_score = sub_df.apply(word_overlap_score, axis=1)
            total = contain_score.add(overlap_score)
            # スコア順
            order = total.sort_values(ascending=False).index
            # sub_df.index は元DataFrameのラベル。ラベル→候補配列内の位置を作る
            index_to_pos = {label: pos for pos, label in enumerate(sub_df.index)}
            ordered_candidate_indices = [candidate_indices[index_to_pos[label]] for label in order if label in index_to_pos][:top_k]
            logger.debug(f"⏱ rerank_results took {(time.time()-start_time)*1000:.1f} ms")
            return ordered_candidate_indices
        except Exception as e:
            st.error(f"リランキングエラー: {str(e)}")
            return candidate_indices[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 50, vector_weight: float = 0.6) -> List[Dict]:
        """ハイブリッド検索を実行（検索を並列化、結合処理を効率化）。"""
        total_start = time.time()
        try:
            logger.info(f"🔍 クエリ: '{query}' (top_k={top_k})")

            # 並列実行（同期I/OのためThreadPoolで十分）
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_vec = ex.submit(self.vector_search, query, top_k * 2)
                f_bm25 = ex.submit(self.keyword_search, query, top_k * 2)
                f_tfidf = ex.submit(self.tfidf_search, query, top_k * 2)
                vector_results = f_vec.result()
                bm25_results = f_bm25.result()
                tfidf_results = f_tfidf.result()
            
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
            
            # スコア順にソート（NumPyで高速化）
            import numpy as _np
            if all_candidates:
                idxs = _np.fromiter(all_candidates.keys(), dtype=_np.int64)
                vals = _np.fromiter(all_candidates.values(), dtype=_np.float32)
                order = _np.argsort(vals)[::-1]
                candidate_indices = idxs[order][: top_k * 2].tolist()
            else:
                candidate_indices = []
            
            logger.info(f"🔄 リランキング前の候補数: {len(candidate_indices)}")
            
            final_top_k = min(top_k, 40)  
            reranked_indices = self.rerank_results(query, candidate_indices, final_top_k)
            
            logger.info(f"✅ リランキング後の結果数: {len(reranked_indices)} (最大{final_top_k})")
            
            # --- ここからグルーピング処理 ---
            results = []
            processed_groups = set() # 処理済みのgroup_codeを記録
            bunya_counts = {}
            
            for idx in reranked_indices:
                item = self.df.iloc[idx]
                group_code = item.get('group_code')

                if group_code not in processed_groups:
                    bunya = item['bunya_name']
                    bunya_counts[bunya] = bunya_counts.get(bunya, 0) + 1
                    
                    results.append({
                        'koumoku_name_full': item['koumoku_name_full'],
                        'bunya_name': item['bunya_name'],
                        'chuubunrui_name': item['chuubunrui_name'],
                        'shoubunrui_name': item['shoubunrui_name'],
                        'koumoku_code': item.get('koumoku_code', ''),
                        'group_code': group_code,
                        'score': all_candidates.get(idx, 0)
                    })
                    
                    processed_groups.add(group_code)

            logger.info(f"⚙️ グルーピング後の最終結果数: {len(results)}件")
            # --- グルーピング処理ここまで ---
            
            logger.info(f"📈 最終結果の分野分布: {dict(bunya_counts)}")
            logger.info(f"🎯 検索完了: {len(results)}件の指標を返却")
            
            logger.debug(f"⏱ hybrid_search total took {(time.time()-total_start)*1000:.1f} ms")
            return results
            
        except Exception as e:
            st.error(f"ハイブリッド検索エラー: {str(e)}")
            return []

# グローバルインスタンス
retriever = HybridRetriever() 