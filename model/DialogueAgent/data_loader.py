# data_loader.py

import json
from transformers import GPT2Tokenizer

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def load_dataset(self):
        # Open and read the JSON file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data

    def prepare_input_data(self, data):
        inputs, responses = [], []

        for dialog in data:
            character = dialog["Character"]
            question = dialog["Question"]
            response = dialog["Response"]

            # Format the dialogue
            formatted_dialog = f'Character: {character}\nQuestion: {question}\nResponse: {response}'

            # Encode the dialogue
            encoded_dialog = self.tokenizer.encode_plus(formatted_dialog, return_tensors='pt')

            # Transform to the format that model needs
            inputs.append(encoded_dialog['input_ids'])
            responses.append(encoded_dialog['attention_mask'])

        return inputs, responses

