from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')

# 取前 149 筆資料
narrative_column = read_data['2022敘述'][:149]  # 文本敘述
gross_margin_column = read_data['毛利率Label'][:149]  # 毛利率標籤
roa_column = read_data['ROA_Label'][:149]  # ROA標籤

# TF 向量化
tf_vectorizer = tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english", lowercase=True)
tf_vectors = tf_vectorizer.fit_transform(narrative_column)  # 使用前 149 筆資料進行 TF 向量化
print(f"TF Vector Shape: {tf_vectors.shape}")

# LDA 降維
lda_model = LDA(n_components=100, random_state=1)
lda_vectors = lda_model.fit_transform(tf_vectors)
print(f"LDA Vector Shape: {lda_vectors.shape}")

# 轉換標籤為數值型（假設有三個類別：G, D, B）
gross_margin_column = gross_margin_column.map({'G': 0, 'D': 1, 'B': 2})
roa_column = roa_column.map({'G': 0, 'D': 1, 'B': 2})

# 訓練毛利率標籤預測模型
X_train, X_test, y_train, y_test = train_test_split(lda_vectors, gross_margin_column, test_size=0.1, random_state=42)

# 訓練 ROA 標籤預測模型
X_train_roa, X_test_roa, y_train_roa, y_test_roa = train_test_split(lda_vectors, roa_column, test_size=0.1, random_state=42)

# 使用不同的 Naive Bayes 模型進行訓練和評估
models = {
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB()
}

# 訓練並評估模型
for name, model in models.items():
    print(f"\n{name} - 毛利率標籤預測報告：")
    # 毛利率標籤
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"{name} - ROA標籤預測報告：")
    # ROA 標籤
    model.fit(X_train_roa, y_train_roa)
    y_pred_roa = model.predict(X_test_roa)
    print(classification_report(y_test_roa, y_pred_roa))
