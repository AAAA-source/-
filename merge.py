import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')

# 取前 149 筆資料
narrative_column = read_data['2022敘述'][:149]  # 文本敘述
gross_margin_column = read_data['毛利率Label'][:149]  # 毛利率標籤
roa_column = read_data['ROA_Label'][:149]  # ROA標籤

# 轉換標籤為數值型
label_mapping = {'G': 0, 'D': 1, 'B': 2}
gross_margin_column = gross_margin_column.map(label_mapping)
roa_column = roa_column.map(label_mapping)

# TF-IDF 向量化
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english', lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column)

# Multi-hot 向量化
multihot_vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words='english', lowercase=True, binary=True)
multihot_vectors = multihot_vectorizer.fit_transform(narrative_column)


# 切割資料成 9:1，保持類別均衡
def split_data_balanced(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1 , random_state=42, stratify=y  # 使用 stratify 保持類別分佈
    )
    return X_train, X_test, y_train, y_test

# 對毛利率進行切割
X_train_gross, X_test_gross, y_train_gross, y_test_gross = split_data_balanced(tfidf_vectors, gross_margin_column)

# 建立 SVM 模型
svm_model = SVC(kernel='linear', C=1, gamma='auto', probability=True)  # 需要 probability=True 來獲取預測概率

# 使用 TF-IDF 訓練 SVM
svm_model.fit(X_train_gross, y_train_gross)

# SVM 預測概率
svm_probabilities = svm_model.predict_proba(X_test_gross)

# 建立 Bernoulli Naive Bayes 模型
bernoulli_nb_model = BernoulliNB()

# 使用 Multi-hot Vectors 訓練 Bernoulli Naive Bayes
bernoulli_nb_model.fit(X_train_gross, y_train_gross)

# Bernoulli Naive Bayes 預測概率
bernoulli_nb_probabilities = bernoulli_nb_model.predict_proba(X_test_gross)

# 定義每個類別的權重
svm_weight = { 'G' : 0.8 , 'D': 0.55 , 'B': 0.8 }
bernoulli_nb_weight = {'G': 0 , 'D': 0 , 'B': 0}

# 計算每個類別的加權信心分數
total_score = {}
for i, label in enumerate(['G', 'D', 'B']):
    # 計算每個類別的加權信心分數（取預測概率並與權重相乘）
    total_score[label] = (svm_probabilities[:, i] * svm_weight[label]).sum() + \
                         (bernoulli_nb_probabilities[:, i] * bernoulli_nb_weight[label]).sum()

# 顯示加權總分數
print("加權信心總分數 (total_score):", total_score)

# 使用加權信心分數做預測：選擇信心分數最高的類別
final_predictions = [max(total_score, key=total_score.get) for _ in range(len(svm_probabilities))]

# 顯示最終預測結果
print("Final Predictions:", final_predictions)

# 預測的標籤 'G', 'D', 'B' 對應到數字標籤 0, 1, 2
label_reverse_mapping = {'G': 0, 'D': 1, 'B': 2}

# 將 final_predictions 轉換為數字標籤
final_predictions_numeric = [label_reverse_mapping[label] for label in final_predictions]

# 計算混淆矩陣
conf_matrix = confusion_matrix(y_test_gross, final_predictions_numeric)

# 顯示混淆矩陣
print("Confusion Matrix:")
print(conf_matrix)
