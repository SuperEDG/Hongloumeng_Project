import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn import DataParallel
import transformers
import pickle
import sys
import random
from sklearn.model_selection import train_test_split
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from torch.utils.data.dataset import random_split
from dataset import MyDataset

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def load_dataset(logger, train_path):
    """
    加载训练集和验证集
    """
    logger.info("loading training dataset and validating dataset")

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 设定随机种子确保实验的可重复性
    random.seed(42)
    # 打乱数据集
    random.shuffle(input_list)

    # 划分训练集与验证集
    train_size = int(0.9 * len(input_list))
    input_list_train = input_list[:train_size]
    input_list_val = input_list[train_size:]

    # 训练时，输入数据的最大长度为150
    train_dataset = MyDataset(input_list_train, 150)
    val_dataset = MyDataset(input_list_val, 150)

    return train_dataset, val_dataset

def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, log_step, save_model_path):
    model.train()
    device = 0
    # 对于ignore_index的label token不计算梯度
    ignore_index = -100
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            # 梯度积累为4
            gradient_accumulation_steps = 4
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1, epoch + 1, loss.item() * gradient_accumulation_steps, batch_acc, scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(save_model_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)
    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, logger, epoch):
    print(len(validate_dataloader))
    logger.info("start validating")
    model.eval()
    device = 0
    ignore_index = -100
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception


def train(model, logger, train_dataset, validate_dataset, batch_size, num_workers, save_model_path, epochs, lr, eps, log_step):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
    print(len(validate_dataloader))
    gradient_accumulation_steps = 4
    t_total = len(train_dataloader) // gradient_accumulation_steps * epochs
    optimizer = transformers.AdamW(model.parameters(), lr=lr, eps=eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=4000, num_training_steps=t_total
    )

    logger.info('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, log_step=log_step, save_model_path=save_model_path)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            logger=logger, epoch=epoch)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
            model_path = join(save_model_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word

def main(train_path,batch_size, warmup_steps, log_path, vocab_path, save_model_path, pretrained_model, model_config, num_workers, epochs, lr, eps, log_step):
    device = 'cuda:0'
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    if batch_size < 2048 and warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    # 创建日志对象
    logger = create_logger(log_path)
    # 当用户使用GPU,并且GPU可用时
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    logger.info('using device:{}'.format(device))

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id

    # 创建模型的输出目录
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    # 创建模型
    if pretrained_model:  # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 并行训练模型
    if cuda and torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset, validate_dataset = load_dataset(logger, train_path)
    train(model, logger, train_dataset, validate_dataset, batch_size, num_workers, save_model_path, epochs, lr, eps, log_step)



if __name__ == "__main__":
    train_path = "/content/train.pkl"
    batch_size = 4
    warmup_steps = 4000
    log_path = "/content/train.log"
    vocab_path = "/content/vocab.txt"
    save_model_path = "/content/model"
    pretrained_model = None
    model_config = "/content/config.json"
    num_workers = 0
    epochs = 2
    lr = 2.6e-5
    eps = 1.0e-09
    log_step = 1

    main(train_path, batch_size, warmup_steps, log_path, vocab_path, save_model_path, pretrained_model, model_config, num_workers, epochs, lr, eps, log_step)