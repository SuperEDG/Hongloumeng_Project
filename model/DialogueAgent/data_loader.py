import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import pandas as pd

class DialogueDataset(Dataset):
    def __init__(self, characters, questions, responses, tokenizer, max_length):
        self.characters = characters
        self.questions = questions
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        character = self.characters[idx]
        question = self.questions[idx]
        response = self.responses[idx]
        
        inputs = f"Character: {character}\nQuestion: {question}\nResponse: {response}"
        encoded = self.tokenizer.encode_plus(
            inputs,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten()
        }

def create_data_loader(characters, questions, responses, tokenizer, max_length, batch_size):
    dataset = DialogueDataset(
        characters,
        questions,
        responses,
        tokenizer,
        max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size
    )

# 加载数据
data = pd.read_csv('dialogues.csv')

# 初始化分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 创建DataLoader
data_loader = create_data_loader(
    data['Character'].tolist(),
    data['Question'].tolist(),
    data['Response'].tolist(),
    tokenizer,
    max_length=128,
    batch_size=32
)
