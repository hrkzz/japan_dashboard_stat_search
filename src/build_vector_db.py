#!/usr/bin/env python3
"""
総務省統計局 社会・人口統計体系データから
ベクトルデータベースを構築するスクリプト

使用方法:
    python build_vector_db.py
"""

import pandas as pd
import faiss
import numpy as np
import os
import json
from datetime import datetime
import argparse
import joblib
from encoder import embedding_config

def verify_api_setup():
    """API設定を確認"""
    try:
        test_embedding = embedding_config.get_single_embedding("テスト")
        if test_embedding.size == 0:
            raise ValueError("エンベディング生成に失敗しました")
        print("✅ API設定確認完了")
        return True
    except Exception as e:
        print(f"❌ API設定エラー: {str(e)}")
        print("環境変数 OPENAI_API_KEY / GEMINI_API_KEY または OLLAMA_BASE_URL を設定してください")
        return False

def load_and_preprocess_data():
    """CSVデータを読み込み、前処理を行う"""
    try:
        print("📂 データファイルを読み込み中...")
        
        # CSVファイルの読み込み
        mst_df = pd.read_csv('../data/social_demographic_stat_mst_koumoku.csv')
        def_df = pd.read_csv('../data/social_demographic_stat_def_koumoku.csv')
        
        print(f"   マスターデータ: {len(mst_df):,}行")
        print(f"   定義データ: {len(def_df):,}行")
        
        # PowerBIダッシュボード用に必要な列を選択
        mst_selected = mst_df[['koumoku_code', 'koumoku_name', 'koumoku_name_full', 'bunya_name', 'chuubunrui_name', 'shoubunrui_name', 'stat_name']].copy()
        def_selected = def_df[['koumoku_code', 'definition']].copy()
        
        # 定義データをマージ（LEFT JOIN）
        df = mst_selected.merge(def_selected, on='koumoku_code', how='left')
        
        # NaN値の処理
        df['definition'] = df['definition'].fillna('')
        
        # 検索用テキストを作成
        print("🔤 検索用テキストを生成中...")
        df['search_text'] = df.apply(lambda row: f"{row['koumoku_name_full']} {row['bunya_name']} {row['chuubunrui_name']} {row['shoubunrui_name']} {row['definition']} {row['stat_name']}", axis=1)
        
        print(f"✅ データ前処理完了: {len(df):,}件")
        return df
        
    except FileNotFoundError as e:
        print(f"❌ ファイルが見つかりません: {str(e)}")
        print("   data/ディレクトリにCSVファイルがあることを確認してください")
        return None
    except Exception as e:
        print(f"❌ データ読み込みエラー: {str(e)}")
        return None

def clean_text(text):
    """テキストをクリーニング"""
    if pd.isna(text) or text is None:
        return "統計指標"
    
    # 文字列に変換
    text = str(text).strip()
    
    # 空文字の場合はデフォルト値
    if len(text) == 0:
        return "統計指標"
    
    return text.strip()

def create_embeddings(texts, batch_size=100):
    """テキストリストをEmbeddingに変換"""
    print(f"🧠 Embeddingを生成中... ({len(texts):,}件のテキスト)")
    
    # テキストをクリーニング
    print("   テキストをクリーニング中...")
    cleaned_texts = [clean_text(text) for text in texts]
    
    # 空のテキストがないかチェック
    empty_count = sum(1 for text in cleaned_texts if not text or len(text.strip()) == 0)
    if empty_count > 0:
        print(f"   警告: {empty_count}件の空テキストをデフォルト値で置換しました")
    
    embeddings = []
    
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        print(f"   進捗: {i + len(batch):,} / {len(cleaned_texts):,}")
        
        # バッチ内のテキストを再度検証
        valid_batch = []
        for text in batch:
            if text and len(text.strip()) > 0:
                valid_batch.append(text)
            else:
                valid_batch.append("統計指標")
        
        try:
            batch_embeddings_array = embedding_config.get_embeddings(valid_batch)
            if batch_embeddings_array.size > 0:
                embeddings.extend(batch_embeddings_array.tolist())
            else:
                print(f"❌ バッチ {i//batch_size + 1}でエンベディング生成に失敗")
                return None
            
        except Exception as e:
            print(f"❌ Embedding生成エラー (バッチ {i//batch_size + 1}): {str(e)}")
            return None
    
    print("✅ Embedding生成完了")
    return np.array(embeddings, dtype=np.float32)

def build_faiss_index(embeddings):
    """FAISSインデックスを構築"""
    print("🔍 FAISSインデックスを構築中...")
    
    # インデックスの次元数
    dimension = embeddings.shape[1]
    print(f"   次元数: {dimension}")
    
    # L2正規化
    faiss.normalize_L2(embeddings)
    
    # IndexFlatIPを使用（内積による類似度検索）
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"✅ FAISSインデックス構築完了: {index.ntotal:,}件")
    return index

def save_database(df, faiss_index, output_dir='../vector_db', bm25=None, tfidf_vectorizer=None, tfidf_matrix=None):
    """データベースをファイルに保存"""
    print(f"💾 データベースを保存中... ({output_dir})")
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # DataFrame保存
    print("   DataFrameを保存中...")
    df.to_parquet(f"{output_dir}/processed_data.parquet", index=False)
    
    # FAISSインデックス保存
    print("   FAISSインデックスを保存中...")
    faiss.write_index(faiss_index, f"{output_dir}/faiss_index.bin")
    
    # メタデータ保存
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_records': len(df),
        'embedding_model': embedding_config.embedding_model or embedding_config._get_embedding_model(),
        'vector_dimension': faiss_index.d
    }
    
    with open(f"{output_dir}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    # 追加: BM25/TF-IDF を永続化
    if bm25 is not None:
        print("   BM25インデックスを保存中...")
        joblib.dump(bm25, f"{output_dir}/bm25.joblib")
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        print("   TF-IDFベクタイザ/行列を保存中...")
        joblib.dump({"vectorizer": tfidf_vectorizer, "matrix": tfidf_matrix}, f"{output_dir}/tfidf.joblib")
    
    print("✅ データベース保存完了")

def main():
    parser = argparse.ArgumentParser(description='統計指標ベクトルデータベース構築')
    parser.add_argument('--batch-size', type=int, default=100, help='Embedding生成のバッチサイズ')
    parser.add_argument('--output-dir', default='../vector_db', help='出力ディレクトリ')
    args = parser.parse_args()
    
    print("🚀 ベクトルデータベース構築を開始...")
    print("=" * 50)
    
    # API設定の確認
    if not verify_api_setup():
        return
    
    # データの読み込み・前処理
    df = load_and_preprocess_data()
    if df is None:
        return
    
    # Embeddingの生成
    search_texts = df['search_text'].tolist()
    embeddings = create_embeddings(search_texts, args.batch_size)
    if embeddings is None:
        return
    
    # FAISSインデックスの構築
    faiss_index = build_faiss_index(embeddings)
    
    # 追加: BM25/TF-IDF を事前計算
    try:
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("   BM25/TF-IDF インデックスを構築中...")
        tokenized_texts = [text.split() for text in search_texts]
        bm25 = BM25Okapi(tokenized_texts)
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words=None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(search_texts)
    except Exception as e:
        print(f"⚠️ BM25/TF-IDF 構築に失敗: {e}")
        bm25, tfidf_vectorizer, tfidf_matrix = None, None, None

    # データベースの保存（BM25/TF-IDF も含める）
    save_database(df, faiss_index, args.output_dir, bm25=bm25, tfidf_vectorizer=tfidf_vectorizer, tfidf_matrix=tfidf_matrix)
    
    print("=" * 50)
    print("🎉 ベクトルデータベース構築完了！")
    print(f"   総レコード数: {len(df):,}件")
    print(f"   ベクトル次元: {embeddings.shape[1]}")
    print(f"   保存先: {args.output_dir}")

if __name__ == "__main__":
    main() 