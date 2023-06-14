import os
import json

def process_file(file_path):
    # Read content from the file
    with open(file_path, encoding='utf-8',  mode = 'r') as f:
        content = f.read()
        # Split the content by space
        space_split = content.split()
        traits = []
        # Split each space-separated part by comma
        for part in space_split:
            traits.extend(part.split(','))
        return set(traits)

def main(folder_path, output_path):
    all_traits = set()

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            traits = process_file(file_path)
            all_traits.update(traits)

    # Sort the traits by length
    sorted_traits = sorted(list(all_traits), key=len)

    traits_by_length = {}
    # Group traits by their length
    for trait in sorted_traits:
        length = len(trait)
        if length not in traits_by_length:
            traits_by_length[length] = []
        traits_by_length[length].append(trait)

    # Save the results to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(traits_by_length, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    folder_path = './data/resources/chineseTraits'  # Change this to the desired folder path
    output_path = './data/processed/chineseTraits.json'  # Change this to the desired output path
    main(folder_path, output_path)
