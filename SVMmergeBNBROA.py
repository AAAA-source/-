# SVN+BNB ROA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

# 讀取資料
read_data = pd.read_csv('datasheet.csv', encoding='utf-8')

# 取前 342 筆資料
narrative_column = read_data['2022敘述'][:342]  # 文本敘述
gross_margin_column = read_data['毛利率Label'][:342]  # 毛利率標籤
roa_column = read_data['ROA_Label'][:342]  # ROA標籤

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
        X, y, test_size=0.1, random_state=42, stratify=y  # 使用 stratify 保持類別分佈
    )
    return X_train, X_test, y_train, y_test

# 對 ROA 進行切割
X_train_roa, X_test_roa, y_train_roa, y_test_roa = split_data_balanced(tfidf_vectors, roa_column)

# 建立 SVM 模型
svm_model = SVC(kernel='linear', C=1, gamma='auto', probability=True)  # 需要 probability=True 來獲取預測概率

# 使用 TF-IDF 訓練 SVM
svm_model.fit(X_train_roa, y_train_roa)

# SVM 預測概率
svm_probabilities = svm_model.predict_proba(X_test_roa)

# 建立 Bernoulli Naive Bayes 模型
bernoulli_nb_model = BernoulliNB()

# 使用 Multi-hot Vectors 訓練 Bernoulli Naive Bayes
bernoulli_nb_model.fit(X_train_roa, y_train_roa)

# Bernoulli Naive Bayes 預測概率
bernoulli_nb_probabilities = bernoulli_nb_model.predict_proba(X_test_roa)

# 動態計算模型的權重
def calculate_model_weights(y_true, svm_probs, bernoulli_probs):
    svm_pred = np.argmax(svm_probs, axis=1)
    bernoulli_pred = np.argmax(bernoulli_probs, axis=1)
    svm_accuracy = np.mean(svm_pred == y_true)
    bernoulli_accuracy = np.mean(bernoulli_pred == y_true)
    total_accuracy = svm_accuracy + bernoulli_accuracy
    svm_weight = svm_accuracy / total_accuracy
    bernoulli_weight = bernoulli_accuracy / total_accuracy
    return svm_weight, bernoulli_weight

# 設定置信度門檻
CONFIDENCE_THRESHOLD = 0.6

# 計算模型權重
svm_weight, bernoulli_weight = calculate_model_weights(y_test_roa, svm_probabilities, bernoulli_nb_probabilities)

# 加權投票結合置信度門檻
final_probabilities = np.zeros_like(svm_probabilities)  # 初始化總概率

for i in range(len(svm_probabilities)):  # 遍歷每個樣本
    for j, label in enumerate(['G', 'D', 'B']):  # 遍歷每個類別
        # SVM 和 Bernoulli 分別判斷是否超過置信度門檻，並加權
        if svm_probabilities[i, j] >= CONFIDENCE_THRESHOLD:
            final_probabilities[i, j] += svm_probabilities[i, j] * svm_weight
        if bernoulli_nb_probabilities[i, j] >= CONFIDENCE_THRESHOLD:
            final_probabilities[i, j] += bernoulli_nb_probabilities[i, j] * bernoulli_weight

# 使用加權概率選擇最終預測類別
final_predictions = np.argmax(final_probabilities, axis=1)

# 計算混淆矩陣
conf_matrix = confusion_matrix(y_test_roa, final_predictions)

# 顯示混淆矩陣
print("Confusion Matrix:")
print(conf_matrix)

# 顯示最終預測結果
print("Final Predictions:", final_predictions)

# 可視化混淆矩陣
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# 顯示分類報告
class_report = classification_report(y_test_roa, final_predictions, target_names=['G', 'D', 'B'])
print("Classification Report:")
print(class_report)

# 顯示混淆矩陣圖
plot_confusion_matrix(conf_matrix, ['G', 'D', 'B'])
