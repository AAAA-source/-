import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# 讀取資料
read_data = pd.read_csv('美股公司財報敘述&財務指標.csv', encoding='utf-8')

# 取前 149 筆資料
narrative_column = read_data['2022敘述'][:149]  # 文本敘述
gross_margin_column = read_data['毛利率Label'][:149]  # 毛利率標籤
roa_column = read_data['ROA_Label'][:149]  # ROA標籤

# 轉換標籤為數值型（假設有三個類別：G, D, B）
gross_margin_column = gross_margin_column.map({'G': 0, 'D': 1, 'B': 2})
roa_column = roa_column.map({'G': 0, 'D': 1, 'B': 2})

# 使用 BERT 的 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 自定義 Dataset 類別來處理文本數據
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # 使用 BERT tokenizer 轉換文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 建立訓練與測試資料集
train_texts, val_texts, train_labels, val_labels = train_test_split(narrative_column, gross_margin_column, test_size=0.1, random_state=42)

train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 載入 BERT 模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# 使用 AdamW 優化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 訓練過程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 訓練循環
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        # 移動到 GPU 或 CPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向傳遞
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # 反向傳遞
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item()}')

# 評估模型
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向傳遞
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 取得預測結果
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 顯示預測結果報告
print("毛利率標籤預測報告 (BERT)：")
print(classification_report(true_labels, predictions))
