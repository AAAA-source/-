# SVM+NB 毛利率
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # 改為 Gaussian Naive Bayes

# 读取数据
read_data = pd.read_csv('datasheet.csv', encoding='utf-8')

# 取前 342 条数据
narrative_column = read_data['2022敘述'][:342]  # 文本叙述
gross_margin_column = read_data['毛利率Label'][:342]  # 毛利率标签
roa_column = read_data['ROA_Label'][:342]  # ROA标签

# 转换标签为数值型
label_mapping = {'G': 0, 'D': 1, 'B': 2}
gross_margin_column = gross_margin_column.map(label_mapping)
roa_column = roa_column.map(label_mapping)

# TF-IDF 向量化
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english', lowercase=True)
tfidf_vectors = tfidf_vectorizer.fit_transform(narrative_column)

# Multi-hot 向量化
multihot_vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words='english', lowercase=True, binary=True)
multihot_vectors = multihot_vectorizer.fit_transform(narrative_column)

# 切割数据成 9:1，保持类别均衡
def split_data_balanced(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y  # 使用 stratify 保持类别分布
    )
    return X_train, X_test, y_train, y_test

# 对毛利率进行切割
X_train_gross, X_test_gross, y_train_gross, y_test_gross = split_data_balanced(tfidf_vectors, gross_margin_column)

# 建立 SVM 模型
svm_model = SVC(kernel='linear', C=1, gamma='auto', probability=True)  # 需要 probability=True 来获取预测概率

# 使用 TF-IDF 训练 SVM
svm_model.fit(X_train_gross, y_train_gross)

# SVM 预测概率
svm_probabilities = svm_model.predict_proba(X_test_gross)

# 建立 Gaussian Naive Bayes 模型
gaussian_nb_model = GaussianNB()

# 使用 TF-IDF 训练 Gaussian Naive Bayes
gaussian_nb_model.fit(X_train_gross.toarray(), y_train_gross)  # 注意：GaussianNB 不支持稀疏矩阵，因此需要转换为密集矩阵

# Gaussian Naive Bayes 预测概率
gaussian_nb_probabilities = gaussian_nb_model.predict_proba(X_test_gross.toarray())

# 动态计算模型的权重
def calculate_model_weights(y_true, svm_probs, gaussian_probs):
    svm_pred = np.argmax(svm_probs, axis=1)
    gaussian_pred = np.argmax(gaussian_probs, axis=1)
    svm_accuracy = np.mean(svm_pred == y_true)
    gaussian_accuracy = np.mean(gaussian_pred == y_true)
    total_accuracy = svm_accuracy + gaussian_accuracy
    svm_weight = svm_accuracy / total_accuracy
    gaussian_weight = gaussian_accuracy / total_accuracy
    return svm_weight, gaussian_weight

# 设置置信度门槛
CONFIDENCE_THRESHOLD = 0.6

# 计算模型权重
svm_weight, gaussian_weight = calculate_model_weights(y_test_gross, svm_probabilities, gaussian_nb_probabilities)

# 加权投票结合置信度门槛
final_probabilities = np.zeros_like(svm_probabilities)  # 初始化总概率

for i in range(len(svm_probabilities)):  # 遍历每个样本
    for j, label in enumerate(['G', 'D', 'B']):  # 遍历每个类别
        # SVM 和 Gaussian 分别判断是否超过置信度门槛，并加权
        if svm_probabilities[i, j] >= CONFIDENCE_THRESHOLD:
            final_probabilities[i, j] += svm_probabilities[i, j] * svm_weight
        if gaussian_nb_probabilities[i, j] >= CONFIDENCE_THRESHOLD:
            final_probabilities[i, j] += gaussian_nb_probabilities[i, j] * gaussian_weight

# 使用加权概率选择最终预测类别
final_predictions = np.argmax(final_probabilities, axis=1)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test_gross, final_predictions)

# 显示混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)

# 显示最终预测结果
print("Final Predictions:", final_predictions)

# 可视化混淆矩阵
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# 显示分类报告
class_report = classification_report(y_test_gross, final_predictions, target_names=['G', 'D', 'B'])
print("Classification Report:")
print(class_report)

# 显示混淆矩阵图
plot_confusion_matrix(conf_matrix, ['G', 'D', 'B'])
