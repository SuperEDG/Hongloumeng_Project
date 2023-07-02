# Process Directory README

This directory contains the Python scripts responsible for data cleaning and processing, forming the foundation of this project. 

## Directory Contents

The process directory includes three Python files: 

- `extractDialogues.py`: This is the core script that extracts dialogues and character names from the original text, based on quotation marks. It creates a structured dataset that can be used for further analysis and modeling.

- `chinese_traits_extractor.py`: This script is responsible for building a lexicon of Chinese personality traits. It categorizes collected Chinese trait words and phrases by the number of characters, ranging from one to multiple. This lexicon can be continuously expanded as more trait words/phrases are found.

- `clean_and_store_reviews.py`: This script pre-processes personality analysis articles corresponding to character names. It performs basic cleaning operations and saves the processed data for later stages.

Please note that these scripts are meant to be run sequentially as part of the data processing pipeline. The output of each script serves as the input for the next, forming the processed data in the `data/processed` directory, which can be used for model training in the next stages of the project.
