from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')

# 取前 149 筆資料
narrative_column = read_data['2022敘述']  # 文本敘述
gross_margin_column = read_data['毛利率Label']  # 毛利率標籤
roa_column = read_data['ROA_Label']  # ROA標籤

# 使用 CountVectorizer
count_vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words='english', lowercase=True)
X = count_vectorizer.fit_transform(narrative_column)

# 轉換標籤為數值型（假設有三個類別：G, D, B）
gross_margin_column = gross_margin_column.map({'G': 0, 'D': 1, 'B': 2})
roa_column = roa_column.map({'G': 0, 'D': 1, 'B': 2})

# 訓練毛利率標籤預測模型
X_train, X_test, y_train, y_test = train_test_split(X, gross_margin_column, test_size=0.1, random_state=42)
cnb_gross_margin = ComplementNB()
cnb_gross_margin.fit(X_train, y_train)

# 預測並評估毛利率標籤模型
y_pred_gross_margin = cnb_gross_margin.predict(X_test)
print("毛利率標籤預測報告 (ComplementNB)：")
print(classification_report(y_test, y_pred_gross_margin))

# 混淆矩陣
conf_matrix_gross = confusion_matrix(y_test, y_pred_gross_margin)
print("毛利率標籤混淆矩陣：")
print(conf_matrix_gross)

# 訓練ROA標籤預測模型
X_train, X_test, y_train, y_test = train_test_split(X, roa_column, test_size=0.1, random_state=42)
cnb_roa = ComplementNB()
cnb_roa.fit(X_train, y_train)

# 預測並評估ROA標籤模型
y_pred_roa = cnb_roa.predict(X_test)
print("ROA標籤預測報告 (ComplementNB)：")
print(classification_report(y_test, y_pred_roa))

# 混淆矩陣
conf_matrix_roa = confusion_matrix(y_test, y_pred_roa)
print("ROA標籤混淆矩陣：")
print(conf_matrix_roa)
