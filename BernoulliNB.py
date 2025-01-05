from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')

# 取所有資料
narrative_column = read_data['2022敘述']  # 文本敘述
gross_margin_column = read_data['毛利率Label']  # 毛利率標籤
roa_column = read_data['ROA_Label']  # ROA標籤

# 使用 CountVectorizer 生成詞頻矩陣，並將其轉換為二元特徵 (multihot)
count_vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words='english', lowercase=True, binary=True)
X = count_vectorizer.fit_transform(narrative_column)

# 轉換標籤為數值型（假設有三個類別：G, D, B）
gross_margin_column = gross_margin_column.map({'G': 0, 'D': 1, 'B': 2})
roa_column = roa_column.map({'G': 0, 'D': 1, 'B': 2})

# 訓練毛利率標籤預測模型
X_train, X_test, y_train, y_test = train_test_split(X, gross_margin_column, test_size=0.1, random_state=42)
bnb_gross_margin = BernoulliNB()
bnb_gross_margin.fit(X_train, y_train)

# 預測並評估毛利率標籤模型
y_pred_gross_margin = bnb_gross_margin.predict(X_test)
print("毛利率標籤預測報告 (BernoulliNB)：")
print(classification_report(y_test, y_pred_gross_margin))

# 訓練ROA標籤預測模型
X_train, X_test, y_train, y_test = train_test_split(X, roa_column, test_size=0.1, random_state=42)
bnb_roa = BernoulliNB()
bnb_roa.fit(X_train, y_train)

# 預測並評估ROA標籤模型
y_pred_roa = bnb_roa.predict(X_test)
print("ROA標籤預測報告 (BernoulliNB)：")
print(classification_report(y_test, y_pred_roa))