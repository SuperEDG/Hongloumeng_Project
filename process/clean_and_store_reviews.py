import os
import json
import re

def clean_text(text):
    # Remove extra spaces, newlines, and special characters
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    return cleaned_text

def main(folder_path, output_file):
    output_data = []

    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        if os.path.isdir(subfolder_path):
            name = re.sub(r'^\d*_', '', subfolder_name)  # Remove digits and underscore from folder name

            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(subfolder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        cleaned_text = clean_text(text)

                        data = {
                            'Name': name,
                            'Reviews': cleaned_text,
                            'Source': ''  # Leave it empty for adding sources later
                        }
                        output_data.append(data)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    folder_path = './data/resources/DreamOfTheRedChamber'
    output_file = './data/processed/reviews.json' 
    main(folder_path, output_file)