#!/usr/bin/env python
"""
Code adapted from https://github.com/stoensin/BERT_fun/blob/master/augment_data.py
"""

import ast

def data_augmentation(saved_file):
    speakers = []
    contexts = []
    combined_res = []
    # combine N speakers with M contexts to get N*M examples
    with open(saved_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            res = ast.literal_eval(line)
            speakers.append(res['speaker'])
            ctx = res['context']
            ctx_left = ctx[:res["istart"]]
            ctx_right = ctx[res["iend"]:]
            contexts.append([ctx_left, ctx_right, res["istart"]])

    uid = 0
    len_truncate = 128
    count = 0  # Control the number of generated data
    for speaker in speakers:
        for ctx_left, ctx_right, istart in contexts:
            ### do not create new sentence for contexts without original speaker
            if istart == -1: continue
            try:
                new_ctx = ctx_left + speaker + ctx_right
                new_iend = istart + len(speaker)
                new_speaker = speaker

                # truncate the input if the sentence is longer than 128 words
                if len(new_ctx) > len_truncate:
                    truncated_ctx = new_ctx[-len_truncate:]
                    # if the speaker is in the truncated_ctx
                    if (len(new_ctx) - istart) < len_truncate:
                        istart = istart - (len(new_ctx) - len_truncate)
                        new_iend = istart + len(speaker)
                        new_speaker = truncated_ctx[istart:new_iend]
                    else:  # if the speaker is not in the truncated_ctx
                        istart = -1
                        new_iend = 0
                        new_speaker = None
                    new_ctx = truncated_ctx
                res = {'uid': uid,
                       'context': new_ctx,
                       'speaker': new_speaker,
                       'istart': istart,
                       'iend': new_iend}
                combined_res.append(res)
                uid = uid + 1
                count += 1
                if count >= 500000:  # Control the number of generated data to 500,000
                    return combined_res
            except:
                continue
    return combined_res


if __name__ == "__main__":
    saved_file = r"C:\Users\Admin\Desktop\Project\Hongloumeng_Project\data\processed\Component2-NER\label_honglou.txt"
    speakers = data_augmentation(saved_file)
    new_file = r"C:\Users\Admin\Desktop\Project\Hongloumeng_Project\data\processed\Component2-NER\label_honglou_new.txt"  # New file name
    # Write the generated data to a new file
    with open(new_file, "w", encoding="utf-8") as f:
        for res in speakers:
            line = str(res) + "\n"
            f.write(line)
