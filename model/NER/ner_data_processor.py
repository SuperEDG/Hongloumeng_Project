import os
from transformers import BertTokenizer

class NERDataProcessor:
    def __init__(self, tokenizer, label_map):
        # Initialize the NERDataProcessor with a tokenizer and a label_map
        self.tokenizer = tokenizer
        self.label_map = label_map

    def read_bio_data(self, file_path):
        # Read BIO format data from a file and return tokens and labels
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        tokens, labels = [], []
        sentence, tags = [], []

        # Iterate through the lines in the file
        for line in lines:
            # If the line is empty, it indicates the end of a sentence
            if line.strip() == "":
                if sentence and tags:
                    tokens.append(sentence)
                    labels.append(tags)
                sentence, tags = [], []
            else:
                # Split the line into token and label
                token, label = line.strip().split()
                sentence.append(token)
                tags.append(label)

        return tokens, labels

    def tokenize_and_prepare_data(self, tokens, labels, max_length=128):
        # Tokenize and prepare data for BERT+BiLSTM+CRF model
        input_ids, attention_masks, tag_ids = [], [], []

        # Iterate through the tokens and labels
        for token_list, label_list in zip(tokens, labels):
            # Encode the tokens using BERT tokenizer
            inputs = self.tokenizer.encode_plus(token_list, is_split_into_words=True, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)
            input_ids.append(inputs["input_ids"])
            attention_masks.append(inputs["attention_mask"])

            # Convert labels to tag_ids and add padding
            padding_len = max_length - len(label_list) - 2
            tag_ids.append([0] + [self.label_map[label] for label in label_list] + [0] * padding_len)

        return input_ids, attention_masks, tag_ids

# Load tokenizer and define label_map
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
label_map = {"B-ORG": 1, "I-ORG": 2, "O": 0}

# Initialize the NERDataProcessor
data_processor = NERDataProcessor(tokenizer, label_map)

# Read data
train_file = "../data/processed/train_bio.txt"
tokens, labels = data_processor.read_bio_data(train_file)

# Tokenize and prepare data
input_ids, attention_masks, tag_ids = data_processor.tokenize_and_prepare_data(tokens, labels)

# Example of input data format
print("Example of input data format:")
print("Tokenized sentence:", tokens[0])
print("Corresponding labels:", labels[0])
