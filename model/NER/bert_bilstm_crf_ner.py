import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from ner_data_processor import NERDataProcessor
import numpy as np
from sklearn.metrics import classification_report

# Define the BERT+BiLSTM+CRF model
class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model, num_tags, lstm_hidden_dim, device):
        super(BertBiLSTMCRF, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, lstm_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.device = device

    def forward(self, input_ids, attention_masks, tags=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_masks)
        lstm_outputs, _ = self.lstm(bert_outputs[0])
        logits = self.fc(lstm_outputs)

        if tags is not None:
            loss = -1 * self.crf(torch.log_softmax(logits, dim=2), tags, mask=attention_masks.bool(), reduction="mean")
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_masks.bool())
            return predictions

def main(train_file, val_file, batch_size, epochs, model_weights_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and define label_map
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    label_map = {"B-ORG": 1, "I-ORG": 2, "O": 0}
    num_tags = len(label_map)

    # Initialize the NERDataProcessor
    data_processor = NERDataProcessor(tokenizer, label_map)

    # Read training data
    tokens, labels = data_processor.read_bio_data(train_file)

    # Tokenize and prepare training data
    input_ids, attention_masks, tag_ids = data_processor.tokenize_and_prepare_data(tokens, labels)

    # Convert training data to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    tag_ids = torch.tensor(tag_ids, dtype=torch.long)

    # Create DataLoader for training data
    train_data = TensorDataset(input_ids, attention_masks, tag_ids)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Read validation data
    val_tokens, val_labels = data_processor.read_bio_data(val_file)

    # Tokenize and prepare validation data
    val_input_ids, val_attention_masks, val_tag_ids = data_processor.tokenize_and_prepare_data(val_tokens, val_labels)

    # Convert validation data to tensors
    val_input_ids = torch.tensor(val_input_ids, dtype=torch.long)
    val_attention_masks = torch.tensor(val_attention_masks, dtype=torch.long)
    val_tag_ids = torch.tensor(val_tag_ids, dtype=torch.long)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_input_ids, val_attention_masks, val_tag_ids)
    val_dataloader = DataLoader(val_data, batch_size=32)

    # Load BERT model and initialize the BERT+BiLSTM+CRF model
    bert_model = BertModel.from_pretrained("bert-base-chinese").to(device)
    lstm_hidden_dim = 128
    model = BertBiLSTMCRF(bert_model, num_tags, lstm_hidden_dim, device).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Define evaluation function
    def evaluate(model, dataloader, device):
        model.eval()
        true_labels, pred_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_masks, tag_ids = tuple(t.to(device) for t in batch)
                predictions = model(input_ids, attention_masks)

                for i, pred in enumerate(predictions):
                    cur_true_labels = tag_ids[i].cpu().tolist()
                    cur_pred_labels = pred
                    cur_attention_mask = attention_masks[i].cpu().tolist()

                    # Remove padding tokens
                    cur_true_labels = [label for idx, label in enumerate(cur_true_labels) if cur_attention_mask[idx] != 0]
                    cur_pred_labels = [label for idx, label in enumerate(cur_pred_labels) if cur_attention_mask[idx] != 0]

                    true_labels.extend(cur_true_labels)
                    pred_labels.extend(cur_pred_labels)

        return true_labels, pred_labels

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids, attention_masks, tag_ids = tuple(t.to(device) for t in batch)

            model.zero_grad()
            loss = model(input_ids, attention_masks, tags=tag_ids)
            loss.backward()

            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

        # Evaluate on validation data
        true_labels, pred_labels = evaluate(model, val_dataloader, device)
        print(classification_report(true_labels, pred_labels, target_names=label_map.keys()))

    print("Training complete!")

    # Save model weights
    torch.save(model.state_dict(), model_weights_path)

if __name__ == "__main__":
    train_file = "../data/processed/train_bio.txt"
    val_file = "../data/processed/val_bio.txt"
    batch_size = 32
    epochs = 3
    model_weights_path = "bert_bilstm_crf_model_weights.pth"
    main(train_file, val_file, batch_size, epochs, model_weights_path)