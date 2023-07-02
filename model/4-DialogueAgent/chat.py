import transformers
import torch
import os
import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
from os.path import join, exists
from itertools import zip_longest, chain
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0

# 在这里定义你的参数
device = '0'
temperature = 1
topk = 8
topp = 0
log_path = 'data/interact.log'
vocab_path = '/content/vocab.txt'
model_path = '/content/model/epoch2'
save_samples_path = "sample/"
repetition_penalty = 1.0
max_len = 25
max_history_len = 3
no_cuda = False

# ... 你的其他参数

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def main():
    logger = create_logger(log_path)
    cuda = torch.cuda.is_available() and not no_cuda
    device = 'cuda' if cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    if save_samples_path:
        if not os.path.exists(save_samples_path):
            os.makedirs(save_samples_path)
        samples_file = open(save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            role = input("role:")
            text = input("user:")

            if save_samples_path:
                samples_file.write("role:{}\n".format(role))
                samples_file.write("user:{}\n".format(text))

            role_ids = tokenizer.encode(role, add_special_tokens=False)
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(role_ids)
            history.append(text_ids)

            input_ids = [tokenizer.cls_token_id]

            for history_id, history_utr in enumerate(history[-2 * max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)

            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)

            generated = model.generate(input_ids, max_length=max_len, temperature=temperature, top_k=topk, top_p=topp, repetition_penalty=repetition_penalty, do_sample=True)
            generated = generated[:, input_ids.shape[-1]:].tolist()

            for generated_sentence in generated:
                text = tokenizer.decode(generated_sentence, skip_special_tokens=True)
                print("chatbot:" + text)
                if save_samples_path:
                    samples_file.write("chatbot:{}\n".format(text))
        except KeyboardInterrupt:
            if save_samples_path:
                samples_file.close()
            break
