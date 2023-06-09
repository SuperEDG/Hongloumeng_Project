# train.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from data_loader import DataLoader as CustomDataLoader

class CustomDataset(Dataset):
    def __init__(self, character, inputs, responses):
        self.characters = characters
        self.inputs = inputs
        self.responses = responses

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.responses[idx]

def train_model(model, dataloader, optimizer, scheduler, epochs):
    model.train()
    loss_function = CrossEntropyLoss()

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            inputs, responses = data

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs.logits.view(-1, model.config.vocab_size), responses.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    loss_function = CrossEntropyLoss()

    for data in dataloader:
        inputs, responses = data

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs.logits.view(-1, model.config.vocab_size), responses.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Paths to your training and validation data
    training_data_path = 'train_data.json'
    validation_data_path = 'val_data.json'

    # Epochs and learning rate
    epochs = 3
    learning_rate = 0.001

    # Load the model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Instantiate the custom DataLoader
    train_loader = CustomDataLoader(training_data_path)
    val_loader = CustomDataLoader(validation_data_path)

    # Load and prepare input data
    train_data = train_loader.load_dataset()
    train_inputs, train_responses = train_loader.prepare_input_data(train_data)

    val_data = val_loader.load_dataset()
    val_inputs, val_responses = val_loader.prepare_input_data(val_data)

    # Create PyTorch Datasets and DataLoaders
    train_dataset = CustomDataset(train_inputs, train_responses)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = CustomDataset(val_inputs, val_responses)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Set the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set the scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model
    train_model(model, train_dataloader, optimizer, scheduler, epochs)

    # Evaluate the
