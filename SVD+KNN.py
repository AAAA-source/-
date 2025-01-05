import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD


## 前處理
# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')
print(read_data.columns)

# 提取敘述和標籤
narrative_column = read_data['2022敘述']  # 文本敘述
gross_margin_column = read_data['毛利率Label']  # 毛利率標籤
roa_column = read_data['ROA_Label'] # ROA標籤

# TF-IDF 向量化
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english", lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column[:342])  # 目前DATA範圍
print(f"TF-IDF Vector Shape: {tfidf_vectors.shape}")

# TF 向量化
tf_vectorizer = CountVectorizer(min_df=1, stop_words='english')
tf_vectors = tf_vectorizer.fit_transform(narrative_column[:342])  # 目前DATA範圍
print(f"TF Vector Shape: {tf_vectors.shape}")

# LDA 降維
lda_model = LDA(n_components=10, random_state=1)
lda_vectors = lda_model.fit_transform(tf_vectors)
print(f"LDA Vector Shape: {lda_vectors.shape}")

## SVD run
svd_model = TruncatedSVD(n_components = 10)
svd_vectors = svd_model.fit_transform(tfidf_vectors)
print(svd_vectors.shape)
print(svd_model.components_.shape)
print(svd_model.singular_values_)


x_train, x_test, y_train_gross, y_test_gross, y_train_roa, y_test_roa = train_test_split(
    svd_vectors,
    gross_margin_column[0:342],
    roa_column[0:342],
    test_size=0.1,
    random_state=42
)

## gross margin
# KNN 模型訓練
KNN_model = KNeighborsClassifier(n_neighbors=11)
KNN_model.fit(x_train, y_train_roa)

# 預測與評估
gross_margin_predicted_results = KNN_model.predict(x_test)
# 預測測試集並評估
print("\nSVD Gross margin Test Set Evaluation:")
print(metrics.classification_report(y_test_gross, gross_margin_predicted_results))

from sklearn.metrics import confusion_matrix
gross_margin_cm = confusion_matrix(y_test_gross, gross_margin_predicted_results)
print(gross_margin_cm)

## roa
# KNN 模型訓練
KNN_model = KNeighborsClassifier(n_neighbors=11)
KNN_model.fit(x_train, y_train_roa)

# 預測與評估
roa_predicted_results = KNN_model.predict(x_test)
# 預測測試集並評估
print("\nSVD ROA Test Set Evaluation:")
print(metrics.classification_report(y_test_roa, roa_predicted_results))

from sklearn.metrics import confusion_matrix
roa_cm = confusion_matrix(y_test_roa, roa_predicted_results)
print(roa_cm)

## LDA run

x_train, x_test, y_train_gross, y_test_gross, y_train_roa, y_test_roa = train_test_split(
    lda_vectors,
    gross_margin_column[0:342],
    roa_column[0:342],
    test_size=0.1,
    random_state=42
)
## gross margin
# KNN 模型訓練
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(x_train, y_train_gross)

# 預測與評估
gross_margin_predicted_results = KNN_model.predict(x_test)
# 預測測試集並評估
print("\nLDA Gross margin Test Set Evaluation:")
print(metrics.classification_report(y_test_gross, gross_margin_predicted_results))

from sklearn.metrics import confusion_matrix
gross_margin_cm = confusion_matrix(y_test_gross, gross_margin_predicted_results)
print(gross_margin_cm)

## roa
# KNN 模型訓練
KNN_model = KNeighborsClassifier(n_neighbors=11)
KNN_model.fit(x_train, y_train_roa)

# 預測與評估
roa_predicted_results = KNN_model.predict(x_test)
# 預測測試集並評估
print("\nLDA ROA Test Set Evaluation:")
print(metrics.classification_report(y_test_roa, roa_predicted_results))

from sklearn.metrics import confusion_matrix
roa_cm = confusion_matrix(y_test_roa, roa_predicted_results)
print(roa_cm)
