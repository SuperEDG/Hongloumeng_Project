import os
import torch
import numpy as np
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Define a class to process NER data
class NERDataProcessor:
    # Initialize the processor
    def __init__(self, tokenizer, label_map, max_length=128):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    # Function to read the BIO format data
    def read_bio_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        tokens, labels = [], []
        sentence, tags = [], []
        for line in lines:
            if line.strip() == "":
                if sentence and tags:
                    tokens.append(sentence)
                    labels.append(tags)
                sentence, tags = [], []
            else:
                token, label = line.strip().split()
                sentence.append(token)
                tags.append(label)

        return tokens, labels

    # Function to tokenize and prepare the data
    def tokenize_and_prepare_data(self, tokens, labels):
        input_ids, attention_masks, tag_ids = [], [], []

        for token_list, label_list in zip(tokens, labels):
            inputs = self.tokenizer.encode_plus(token_list, is_split_into_words=True, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)

            input_ids.append(inputs["input_ids"])
            attention_masks.append(inputs["attention_mask"])

            subword_lengths = []
            for word in token_list:
                subword_lengths.append(len(self.tokenizer.tokenize(word)))
            expanded_labels = []
            for label, sub_len in zip(label_list, subword_lengths):
                expanded_labels.extend([self.label_map[label]] * sub_len)

            expanded_labels = [self.label_map['O']] + expanded_labels + [self.label_map['O']]

            if len(expanded_labels) < self.max_length:
                expanded_labels = expanded_labels + [0] * (self.max_length - len(expanded_labels))
            else:
                expanded_labels = expanded_labels[:self.max_length]

            tag_ids.append(expanded_labels)

        input_ids = np.array(input_ids, dtype=int)
        attention_masks = np.array(attention_masks, dtype=int)
        tag_ids = np.array(tag_ids, dtype=int)

        return input_ids, attention_masks, tag_ids

# Define the model class
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
            return {"loss": loss}
        else:
            emissions = torch.log_softmax(logits, dim=2)
            predictions = self.crf.decode(emissions, mask=attention_masks.bool())
            return {"emissions": emissions, "predictions": predictions}

# Define the main function
def main(train_file, val_file, batch_size, epochs, model_weights_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and define label_map
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    label_map = {"B-ORG": 1, "I-ORG": 2, "O": 0}
    num_tags = len(label_map)

    # Initialize the NERDataProcessor
    data_processor = NERDataProcessor(tokenizer, label_map)

    # Load BERT model and initialize the BERT+BiLSTM+CRF model
    bert_model = BertModel.from_pretrained("bert-base-chinese")
    lstm_hidden_dim = 256
    model = BertBiLSTMCRF(bert_model, num_tags, lstm_hidden_dim, device).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

    tokens, labels = data_processor.read_bio_data(train_file)
    input_ids, attention_masks, tag_ids = data_processor.tokenize_and_prepare_data(tokens, labels)
    train_data = TensorDataset(torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    val_tokens, val_labels = data_processor.read_bio_data(val_file)
    val_input_ids, val_attention_masks, val_tag_ids = data_processor.tokenize_and_prepare_data(val_tokens, val_labels)
    val_data = TensorDataset(torch.tensor(val_input_ids, dtype=torch.long), torch.tensor(val_attention_masks, dtype=torch.long), torch.tensor(val_tag_ids, dtype=torch.long))
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_tags = batch
            model.zero_grad()
            outputs = model(b_input_ids, b_attention_masks, tags=b_tags)
            loss = outputs["loss"]
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print("Epoch: {}, Training Loss: {}".format(epoch + 1, avg_train_loss))

        # Evaluate on validation data
        model.eval()
        val_predictions, true_labels = [], []

        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_tags = batch
            with torch.no_grad():
                outputs = model(b_input_ids, b_attention_masks)

            # Remove padding part in the predictions
            for pred, mask in zip(outputs["predictions"], b_attention_masks):
                pred = [p for p, m in zip(pred, mask) if m==1]
                val_predictions.extend(pred)

            # Remove padding part in the true labels
            b_tags = b_tags.view(-1).cpu().numpy().tolist()
            true_labels_batch = [tag for tag, mask in zip(b_tags, b_attention_masks.view(-1)) if mask==1]
            true_labels.extend(true_labels_batch)

        print("Validation Accuracy: {}".format(classification_report(true_labels, val_predictions)))

    # Save the model weights
    torch.save(model.state_dict(), model_weights_path)


# Execute the main function if the script is run as the main program
if __name__ == "__main__":
    train_file = "/content/train_bio.txt"
    val_file = "/content/val_bio.txt"
    batch_size = 32
    epochs = 5
    model_weights_path = "bert_bilstm_crf_model_weights.pth"
    main(train_file, val_file, batch_size, epochs, model_weights_path)