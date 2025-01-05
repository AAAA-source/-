import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')
narrative_column = read_data['2022敘述']  # 敘述
gross_margin_column = read_data['毛利率Label']  # 毛利率標籤
roa_column = read_data['ROA_Label']  # ROA 標籤
# 轉TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english", lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column[0:144])

X_train, X_test, y_train_gross, y_test_gross, y_train_roa, y_test_roa = train_test_split(
    tfidf_vectors,
    gross_margin_column[0:144],
    roa_column[0:144],
    test_size=0.3,
    random_state=42
)
class RocchioClassifier:
    def __init__(self):
        self.class_centroids = {}
    def fit(self, X_train, y_train):
        #算centroid
        for label in np.unique(y_train):
            label_vectors = X_train[y_train == label]
            centroid = np.mean(label_vectors.toarray(), axis=0)
            self.class_centroids[label] = centroid
    def classify(self, doc_vector): #算centroid與testing data的距離
        doc_vector_dense = doc_vector.toarray().flatten()
        similarities = {
            label: cosine_similarity(doc_vector_dense.reshape(1, -1), centroid.reshape(1, -1))[0][0]
            for label, centroid in self.class_centroids.items()
        }
        return max(similarities, key=similarities.get)
    def predict(self, X_test): #預測Xtest的分類
        return [self.classify(doc_vector) for doc_vector in X_test]
rocchio_gross = RocchioClassifier()
rocchio_gross.fit(X_train, y_train_gross)
predictions_gross_rocchio = rocchio_gross.predict(X_test)
rocchio_roa = RocchioClassifier()
rocchio_roa.fit(X_train, y_train_roa)
predictions_roa_rocchio = rocchio_roa.predict(X_test)
print("Rocchio 毛利率 Report:")
report_Rocchio_gross = classification_report(y_test_gross, predictions_gross_rocchio)
print(report_Rocchio_gross)
print("Rocchio Roa Report:")
report_Rocchio_roa = classification_report(y_test_roa, predictions_roa_rocchio)
print(report_Rocchio_roa)
