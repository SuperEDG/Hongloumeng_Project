# 4-DialogueAgent Directory README

The 4-DialogueAgent directory contains scripts and models used to train and interact with the dialogue agent. The GPT-2 model used here is based largely on the open-source project [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese).

## Directory Contents

Here are the core files in the 4-DialogueAgent directory:

- `data_loader.py`: This script is responsible for preparing and loading the data in a format that can be understood by the model. It converts the dialogues generated in the earlier steps into inputs for the GPT-2 model.

- `train.py`: This script handles the training process of the GPT-2 model. Using the preprocessed data from `data_loader.py`, it trains the model to generate dialogue consistent with the personality traits of the characters.

- `chat.py`: This script loads the trained GPT-2 model weights and initiates a chat interface. Users can input prompts and the chatbot will respond based on the learned dialogues and personalities.

Please note that the trained weights for the GPT-2 model are not included in this repository due to their size. If you need the weights for the pre-trained model, feel free to contact the project maintainer at `zhaojunzhe_bit@163.com`.

With these scripts and the trained GPT-2 model, you can interact with a chatbot that has learned to mimic the dialogue style and personality traits of characters from the classic Chinese literature.