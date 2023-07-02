# 2-NER Directory README

This directory contains the scripts for Named Entity Recognition (NER) training and prediction, which is the second component of our model. 

## Directory Contents

The 2-NER directory includes several Python files that perform various functions:

- `augment_data.py`: This script enhances the data by swapping words except for names. For example, "贾宝玉笑道：" can be replaced with "林黛玉笑道：". Both input and output are in `.txt` format.

- `bio_data_processor.py`: This script annotates the original text into the BIO format and divides it into three parts: `train_bio.txt`, `val_bio.txt`, and `test_bio.txt`. For instance, the sentence "旁边宝玉道：" will be annotated as:

```
旁 O
边 O
宝 B-ORG
玉 I-ORG
道 O
： O
```

- `bert_bilstm_crf_ner.py`: This script trains the model using the BERT-BiLSTM-CRF architecture with `batch_size = 32` and `epochs = 5`. Due to the large dataset (500k records), training takes a considerable amount of time. Pre-trained weights can be provided via email at `zhaojunzhe_bit@163.com`.

- `predict.py`: For demonstration purposes, this script loads the trained model and allows you to input sentences for name extraction. For instance, if you input `["那僧笑道：", "杭州的西湖美丽极了,贾宝玉说道"]` into `input_sentences`, it will extract the names.

- `extract_dialogues.py`: This script is used for batch extraction of dialogues. It takes as input a JSON file from the `data/processed/books` directory, which should be named after the book you want to extract from. The output is a `.csv` file.

Here's an example of the output CSV file, formatted using markdown:

```markdown
|    | Name   | Dialogue                   |
|----|--------|----------------------------|
| 1  | 贾宝玉 | 妹妹可曾读书？               |
| 2  | 林黛玉 | 不曾读，只上了一年学，些须认得几个字。|
| 3  | 贾宝玉 | 妹妹尊名是那两个字？             |
| 4  | 林黛玉 | 无字。                     |
| 5  | 贾宝玉 | 我送妹妹一妙字，莫若`颦颦-二字极妙。 |
```

These scripts provide an end-to-end solution for training a NER model, augmenting data, and extracting dialogues from the text.