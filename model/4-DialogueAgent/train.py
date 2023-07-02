import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import random_split

# 加载预训练模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-chinese")
model = GPT2LMHeadModel.from_pretrained("gpt2-chinese")

# 创建数据集和数据加载器
data = pd.read_csv('dialogues.csv')
dataset = DialogueDataset(
    data['Character'].tolist(),
    data['Question'].tolist(),
    data['Response'].tolist(),
    tokenizer,
    max_length=128
)
# 按照一定的比例划分数据集为训练集和验证集
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# 设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
epochs = 5
total_steps = len(train_loader) * epochs
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
model = model.train()
for epoch in range(epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 在验证集上评估模型
    model = model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs[0]
            val_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}，Validation loss: {val_loss/len(val_loader)}')

# 保存训练好的模型
model.save_pretrained('trained_model')
tokenizer.save_pretrained('trained_model')