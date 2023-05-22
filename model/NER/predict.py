import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
from ner_data_processor import NERDataProcessor
from bert_bilstm_crf_ner import BertBiLSTMCRF
import sys
import numpy as np

def load_model(model_weights_path, device):
    # Load tokenizer and define label_map
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    label_map = {"B-ORG": 1, "I-ORG": 2, "O": 0}
    num_tags = len(label_map)

    # Load BERT model and initialize the BERT+BiLSTM+CRF model
    bert_model = BertModel.from_pretrained("bert-base-chinese").to(device)
    lstm_hidden_dim = 128
    model = BertBiLSTMCRF(bert_model, num_tags, lstm_hidden_dim, device).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    return model, tokenizer, label_map

def predict_on_test_data(model, tokenizer, label_map, test_file, device):
    # Initialize the NERDataProcessor
    data_processor = NERDataProcessor(tokenizer, label_map)

    # Read test data
    tokens, labels = data_processor.read_bio_data(test_file)

    # Tokenize and prepare test data
    input_ids, attention_masks, tag_ids = data_processor.tokenize_and_prepare_data(tokens, labels)

    # Convert test data to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    tag_ids = torch.tensor(tag_ids, dtype=torch.long)

    # Create DataLoader for test data
    test_data = TensorDataset(input_ids, attention_masks, tag_ids)
    test_dataloader = DataLoader(test_data, batch_size=32)

    # Predict on test data
    true_labels, pred_labels = [], []
    for batch in test_dataloader:
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

def predict(model, tokenizer, label_map, input_sentence, device):
    model.eval()

    # Tokenize input_sentence
    tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_masks = [1] * len(input_ids)

    # Pad input_ids and attention_masks
    max_len = 128
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_masks = attention_masks + ([0] * padding_length)

    # Convert to tensors
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_masks = torch.tensor([attention_masks], dtype=torch.long).to(device)

    # Make prediction
    with torch.no_grad():
        predictions = model(input_ids, attention_masks)

    # Convert predicted label_ids to labels
    label_ids = predictions[0]
    labels = [key for idx in label_ids for key, value in label_map.items() if value == idx]

    # Extract named entities
    named_entities = []
    entity = ""
    for token, label in zip(tokens, labels):
        if label == "B-ORG":
            if entity:
                named_entities.append(entity)
                entity = ""
            entity = token
        elif label == "I-ORG":
            entity += token
        else:
            if entity:
                named_entities.append(entity)
                entity = ""

    if entity:
        named_entities.append(entity)

    return named_entities

def predict_batch(model, tokenizer, label_map, input_file, output_file, device):
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = f.readlines()

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            sentence = sentence.strip()
            named_entities = predict(model, tokenizer, label_map, sentence, device)
            f.write(f"{sentence}\n")
            f.write("Named entities: " + ", ".join(named_entities) + "\n\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights_path = "bert_bilstm_crf_model_weights.pth"
    model, tokenizer, label_map = load_model(model_weights_path, device)

    test_file = "../data/processed/test_bio.txt"
    true_labels, pred_labels = predict_on_test_data(model, tokenizer, label_map, test_file, device)
    print("True labels:", true_labels)
    print("Predicted labels:", pred_labels)

    """
    input_sentences.txt is a text file containing multiple sentences, with one sentence per line. Each sentence should be on a separate line.
    output_results.txt is the output file that will contain the named entity predictions for each sentence in the input file. The results will be saved in a human-readable format, with the original sentence followed by the list of named entities detected.
    """
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        predict_batch(model, tokenizer, label_map, input_file, output_file, device)
        print(f"Batch prediction completed. Results saved to {output_file}")
    else:
        print("Usage: python predict.py <input_file> <output_file>")
