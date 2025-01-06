import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 原始資料
read_data = pd.read_csv("美股公司財報敘述&財務指標.csv", encoding='utf-8')
stock_column = read_data['公司(股票代碼)']
narrative_column = read_data['2022敘述']
gross_margin_column = read_data['毛利率Label']
roa_column = read_data['ROA_Label']

# TF-IDF 向量化
tfidf_vectorizer = TfidfVectorizer(min_df=2 , max_df = 0.95 , stop_words = "english", lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column)
gross_label = gross_margin_column
roa_label = roa_column

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

def SVM_linear(label, label_name):
    TFIDF_vectors_train, TFIDF_vectors_test, label_train, label_test = train_test_split(tfidf_vectors, label, test_size=0.1, random_state=42)

    for kernel in ['linear', 'rbf', 'poly']:
        SVM_model = SVC(kernel=kernel, C=1, gamma='auto')
        SVM_model.fit(TFIDF_vectors_train, label_train)
        label_predicted = SVM_model.predict(TFIDF_vectors_test)
        print(f"Using {label_name}: Kernel: {kernel}\n{classification_report(label_test, label_predicted, zero_division=0)}")
        print(confusion_matrix(label_test, label_predicted))

# 使用 gross_label 作為標籤
SVM_linear(gross_label, "gross_label")

# 使用 roa_label 作為標籤
SVM_linear(roa_label, "roa_label")

# 使用 KMeans 進行分群
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def K_Means(tfidf_vectors, labels, stock_column, label_name):
    # 分割資料集
    tfidf_vectors_train, tfidf_vectors_test, label_train, label_test, stock_column_train, stock_column_test = train_test_split(
        tfidf_vectors, labels, stock_column, test_size=0.1, random_state=42)

    # 標籤編碼
    label_encoder = LabelEncoder()
    label_train = label_encoder.fit_transform(label_train)
    label_test = label_encoder.transform(label_test)

    # KMeans 分群
    final_model = KMeans(n_clusters=3)
    final_model.fit(tfidf_vectors_train)
    label_predicted = final_model.predict(tfidf_vectors_test)
    
    # 印出結果
    print(f"Using {label_name}:")
    print(f"Classification Report:\n{classification_report(label_test, label_predicted, zero_division=0)}")
    print(f"Confusion Matrix:\n{confusion_matrix(label_test, label_predicted)}")
    
    # 印出分群結果
    order_centroids = final_model.cluster_centers_.argsort()[:, ::-1]
    for i in range(3):
        print(f"\n\nCluster {i}:")
        for ind in order_centroids[i, :10]:
            print(tfidf_vectorizer.get_feature_names_out()[ind])

print("\nKMeans-Using gross_label:")
K_Means(tfidf_vectors, gross_label, stock_column, '毛利率Label')
print("\nKMeans-Using roa_label:")
K_Means(tfidf_vectors, roa_label, stock_column, 'ROA_Label')
