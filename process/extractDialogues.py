import os
from tqdm import tqdm
import json

def get_conversations(sentence):
    '''
    Get the start and end position of the conversation in the input sentence.
    
    Args:
        sentence: str, input sentence containing conversations.
        
    Returns:
        talks: list of dicts, each dict contains the start index, end index, and conversation text.
        contexts: list of str, the context from where one can extract the speaker.
    '''
    end_symbols = ['"', '“', '”']
    istart, iend = -1, -1
    talks = []
    # get the start and end position for conversation
    for i in range(1, len(sentence)): 
        if (not istart == -1) and sentence[i] in end_symbols:
            iend = i
            conversation = {'istart':istart, 'iend':iend, 'talk':sentence[istart+1:iend]}
            talks.append(conversation)
            istart = -1
        if sentence[i-1] in [':', '：'] and sentence[i] in end_symbols:
            istart = i
    # get the context from where one can extract speaker
    contexts = []
    if len(talks):
        for i in range(len(talks)):
            if i == 0: 
                contexts.append(sentence[:talks[i]['istart']])
            else:
                contexts.append(sentence[talks[i-1]['iend']+1:talks[i]['istart']])
        # append the paragraph after the conversation if iend != len(sentence)
        if talks[-1]['iend'] != len(sentence):
            contexts.append(sentence[talks[-1]['iend']+1:])
        else:
            contexts.append(' ')
        # the situation is not considered if the speaker comes after the talk
        for i in range(len(talks)):
            talks[i]['context'] = contexts[i] #+ 'XXXXX' + contexts[i+1]

    return talks, contexts 

def extract_corpus(book_name="Hongloumeng.txt", save_as="Hongloumeng.json"):
    talks_list = []  # Create an empty list to store the conversation dictionaries

    with open(book_name, "r", encoding="utf-8") as fin:
        for line in tqdm(fin.readlines()):
            talks, contexts = get_conversations(line.strip())
            if len(talks) > 0:
                for talk in talks:
                    talks_list.append(talk)  # Add the conversation dictionary to the list

    # Save the talks list to a JSON file
    with open(save_as, "w", encoding="utf-8") as fout:
        formatted_json = json.dumps(talks_list, ensure_ascii=False, indent=2)
        formatted_json = formatted_json.replace("},\n", "},\n\n")  # Add a newline after each comma
        fout.write(formatted_json)

if __name__ == "__main__":
    # Set path
    book_path = "./data/resources/Hongloumeng.txt"
    save_path = "./data/processed/Hongloumeng.json"
    extract_corpus(book_name=book_path, save_as=save_path)
