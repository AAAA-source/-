# simpleRNN ROA
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# simpleRNN
# 讀取資料
read_data = pd.read_csv('datasheet.csv', encoding='utf-8')

# 取前 149 筆資料
narrative_column = read_data['2022敘述'][:342]  # 文本敘述
roa_column = read_data['ROA_Label'][:342]  # ROA標籤

# 轉換標籤為數值型（假設有三個類別：G, D, B）
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
train_texts, val_texts, train_labels, val_labels = train_test_split(narrative_column, roa_column, test_size=0.1, random_state=42)

train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 自定義模型：BERT + SimpleRNN + 全連接層
class BertRNNClassifier(nn.Module):
    def __init__(self, num_labels, hidden_dim=128):
        super(BertRNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn = nn.RNN(
            input_size=self.bert.config.hidden_size,  # BERT 的輸出尺寸
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, num_labels)  # 單向 RNN 所以不需要 *2

        # 凍結 BERT 參數
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = bert_outputs.last_hidden_state  # BERT 每個字譜的輸出

        rnn_output, _ = self.rnn(sequence_output)  # 經過 SimpleRNN
        rnn_output = rnn_output[:, -1, :]  # 只取最後一個時間段的資訊

        x = self.dropout(rnn_output)  # Dropout
        x = self.fc(x)  # 全連接層
        return x

# 初始化模型
model = BertRNNClassifier(num_labels=3)

# 使用 AdamW 優化器（優化所有可訓練參數）
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
)

# 訓練過程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 訓練循環
epochs = 7
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        # 移動到 GPU 或 CPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 清除校騰
        optimizer.zero_grad()

        # 前向傳遞
        outputs = model(input_ids, attention_mask=attention_mask)

        # 計算損失
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # 反向傳遞
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

        # 取得預測結果
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 顯示預測結果報告
print("ROA標籤預測報告 (BERT + SimpleRNN)：")
print(classification_report(true_labels, predictions))

# 新增混淆矩陣及可視化
conf_matrix = confusion_matrix(true_labels, predictions)
print("混淆矩陣：")
print(conf_matrix)

# 計算成功率
success_rate_g = (
    conf_matrix[0, 0] + conf_matrix[0, 1]
) / conf_matrix[0, :].sum() if conf_matrix[0, :].sum() != 0 else 0  # 預測為 G 時，實際為 G 或 D
success_rate_b = (
    conf_matrix[2, 2] + conf_matrix[2, 1]
) / conf_matrix[2, :].sum() if conf_matrix[2, :].sum() != 0 else 0  # 預測為 B 時，實際為 B 或 D

print(f"當預測為 G 時，實際為 G 或 D 的比例：{success_rate_g:.2%}")
print(f"當預測為 B 時，實際為 B 或 D 的比例：{success_rate_b:.2%}")

# 混淆矩陣的可視化
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['G', 'D', 'B'], yticklabels=['G', 'D', 'B'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
