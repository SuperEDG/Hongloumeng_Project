import os
import tqdm
import random

def process_data(input_file, output_file):
    # Open the input file and read the raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    # Initialize an empty list to store the processed data
    processed_data = []

    # Iterate through each line in the raw data, using tqdm to show progress
    for line in tqdm.tqdm(raw_data, desc="Processing data"):
        # Parse the dictionary from the line
        data = eval(line.strip())
        context = data['context']
        speaker = data['speaker']
        istart = data['istart']
        iend = data['iend']

        bio_tags = []
        for i, char in enumerate(context):
            if i == istart:
                bio_tags.append(f"{char} B-ORG")
            elif istart < i < iend:
                bio_tags.append(f"{char} I-ORG")
            else:
                bio_tags.append(f"{char} O")

        processed_data.append("\n".join(bio_tags) + "\n\n")

    # Save the processed data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_data)

def split_data(output_file, train_file, val_file, test_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    with open(output_file, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n\n')

    random.shuffle(data)
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_data) + '\n\n')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(val_data) + '\n\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_data) + '\n\n')

if __name__ == "__main__":
    input_file = "./data/resources/Label_hongloumeng.txt"
    output_file = "./data/processed/bio_data.txt"
    train_file = "./data/processed/train_bio.txt"
    val_file = "./data/processed/val_bio.txt"
    test_file = "./data/processed/test_bio.txt"

    process_data(input_file, output_file)
    print(f"Data has been processed and saved to {output_file}")

    split_data(output_file, train_file, val_file, test_file)
    print(f"Data has been split into {train_file}, {val_file}, and {test_file}")