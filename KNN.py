import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')
narrative_column = read_data['2022敘述']  # 敘述
gross_margin_column = read_data['毛利率Label']  # 毛利率標籤
roa_column = read_data['ROA_Label']  # ROA 標籤
rating_column = read_data['Rating']
# 轉TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english", lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column)
X_train, X_test, y_train_gross, y_test_gross, y_train_roa, y_test_roa = train_test_split(
    tfidf_vectors,
    gross_margin_column,
    roa_column,
    test_size=0.1,
    random_state=42
)

# 加入 KNN
knn_gross = KNeighborsClassifier(n_neighbors=5)  # K = 5
knn_roa = KNeighborsClassifier(n_neighbors=5)
knn_gross.fit(X_train, y_train_gross)
knn_roa.fit(X_train, y_train_roa)
predictions_gross_knn = knn_gross.predict(X_test)
predictions_roa_knn = knn_roa.predict(X_test)
print("KNN 毛利率 Report:")
report_knn_gross = classification_report(y_test_gross, predictions_gross_knn)
print(report_knn_gross)
print("KNN 毛利率 混淆矩陣:")
knngross_confusion_matrix = confusion_matrix(y_test_gross, predictions_gross_knn)
print(knngross_confusion_matrix)
print("KNN ROA Report:")
report_knn_roa = classification_report(y_test_roa, predictions_roa_knn)
print(report_knn_roa)
print("KNN ROA 混淆矩陣")
knnroa_confusion_matrix = confusion_matrix(y_test_roa, predictions_roa_knn)
print(knnroa_confusion_matrix)
