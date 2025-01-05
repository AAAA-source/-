import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import joblib

# 原始資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')
stock_column = read_data['公司(股票代碼)']
narrative_column = read_data['2022敘述']
gross_margin_column = read_data['2023毛利率成長率']
roa_column = read_data['ROA成長率']

# TF-IDF 向量化
tfidf_vectorizer = TfidfVectorizer(min_df=2 , max_df = 0.9 , stop_words = "english", lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column[0:144])
print(tfidf_vectors.shape)

# LDA 降維
LDA_model = LDA(n_components=5, random_state=0)
LDA_vectors = LDA_model.fit_transform(tfidf_vectors)
print(LDA_vectors.shape)
print(LDA_model.components_.shape)

# 匯出檔案
joblib.dump(tfidf_vectors, 'tfidf_vectors.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(LDA_vectors, 'lda_vectors.joblib')
joblib.dump(LDA_model, 'lda_model.joblib')

# 下次執行時可直接讀取檔案
tfidf_vectors = joblib.load('tfidf_vectors.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
LDA_vectors = joblib.load('lda_vectors.joblib')
LDA_model = joblib.load('lda_model.joblib')
