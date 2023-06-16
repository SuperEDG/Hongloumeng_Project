import torch
import numpy as np
from transformers import BertTokenizer
from torchcrf import CRF
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import BertModel

import json
import pandas as pd
from collections import defaultdict

# Define the model
class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model, num_tags, lstm_hidden_dim, device):
        super(BertBiLSTMCRF, self).__init__()
        self.bert = bert_model  # Load the pre-trained BERT model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, lstm_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)  # Bi-directional LSTM layer
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_tags)  # Fully connected layer
        self.crf = CRF(num_tags, batch_first=True)  # CRF layer for sequence labeling
        self.device = device

    def forward(self, input_ids, attention_masks, tags=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_masks)  # Get the BERT output
        lstm_outputs, _ = self.lstm(bert_outputs[0])  # Process the BERT output with LSTM
        logits = self.fc(lstm_outputs)  # Process the LSTM output with the fully connected layer

        if tags is not None:  # If tags are provided, compute the loss
            loss = -1 * self.crf(torch.log_softmax(logits, dim=2), tags, mask=attention_masks.bool(), reduction="mean")
            return {"loss": loss}
        else:  # If no tags are provided, decode the predictions
            emissions = torch.log_softmax(logits, dim=2)
            predictions = self.crf.decode(emissions, mask=attention_masks.bool())
            return {"emissions": emissions, "predictions": predictions}

def main(model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  # Load the tokenizer
    label_map = {"B-ORG": 1, "I-ORG": 2, "O": 0}  # Define the label mapping
    num_tags = len(label_map)  # Get the number of tags
    max_length = 128  # Define the maximum length of input sentences

    bert_model = BertModel.from_pretrained("bert-base-chinese")  # Load the pre-trained BERT model
    lstm_hidden_dim = 256  # Define the hidden dimension of the LSTM
    model = BertBiLSTMCRF(bert_model, num_tags, lstm_hidden_dim, device).to(device)  # Instantiate the model

    model.load_state_dict(torch.load(model_weights_path, map_location=device))  # Load the model weights
    model.eval()  # Set the model to evaluation mode

    return model, tokenizer, label_map

def process_sentence(sentence, model, tokenizer, label_map, max_length):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')  # Tokenize and encode the input sentence
    inputs.to(model.device)  # Move the inputs to the same device as the model
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])  # Get the model predictions

    id2label = {v: k for k, v in label_map.items()}  # Create a mapping from label index to label string
    predictions = outputs["predictions"][0]  # Get the predicted labels

    entities = []
    entity = []
    for i, pred in enumerate(predictions):
        if pred == label_map['O']:  # If the predicted label is 'O', it is outside of any entity
            if entity:
                entities.append("".join(entity))  # Add the current entity to the list of entities
                entity = []  # Reset the entity buffer
            continue
        if id2label[pred][0] == 'B':  # If the predicted label starts with 'B', it is the beginning of a new entity
            if entity:
                entities.append("".join(entity))  # Add the current entity to the list of entities
                entity = []  # Reset the entity buffer
            entity.append(tokenizer.decode(inputs["input_ids"][0][i]))  # Add the current token to the entity buffer
        else:  # If the predicted label starts with 'I', it is inside an entity
            entity.append(tokenizer.decode(inputs["input_ids"][0][i]))  # Add the current token to the entity buffer
    if entity:
        entities.append("".join(entity))  # Add the last entity to the list of entities

    return entities

def extract_name_from_context(context, model, tokenizer, label_map, max_length):
    entities = process_sentence(context, model, tokenizer, label_map, max_length)
    # Assuming there is only one name entity in the "Speaker" context, take the first one as the name of the speaker
    if entities:
        return entities[0]
    else:
        return None

def create_dataframe_from_json(json_file, model, tokenizer, label_map):
    # Load the JSON file
    with open(json_file, "r", encoding='utf-8') as file:
        data = json.load(file)

    # Create a dictionary to store the information of "Speaker" and "Talk"
    dialogue_dict = defaultdict(list)

    # Iterate over each dictionary in the JSON file
    for item in data:
        # Extract the name from the "Speaker" context
        name = extract_name_from_context(item["context"], model, tokenizer, label_map, max_length=128)
        if name is not None:
            # Add the name and dialogue content to the dictionary
            dialogue_dict["Speaker"].append(name)
            dialogue_dict["Talk"].append(item["talk"])

    # Convert the dictionary to a dataframe
    df = pd.DataFrame(dialogue_dict)

    return df


# Execute the main function if the script is run as the main program
if __name__ == "__main__":
    model_weights_path = "/content/bert_bilstm_crf_model_weights.pth"  # Path to the model weights
    model, tokenizer, label_map = main(model_weights_path)  # Load the model, tokenizer, and label map

    json_file = "/content/Hongloumeng.json"  # Path to your json file
    df = create_dataframe_from_json(json_file, model, tokenizer, label_map)  # Create a dataframe from the json file
    print(df)
    csv_file = "dialogue_data.csv"  # Specify the path of the csv file
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')