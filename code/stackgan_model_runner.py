import multiprocessing
from datasets import DatasetLoadTransform
import os
import nltk
import spacy
from stackgan_runner import GAN
from utils import mkdir_p
from configuration import conf, cfg
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
import argparse
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
import tensorflow as tf
import torch
import torch.nn as nn
import nltk
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
from configuration import cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='yaml/coco_eval.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


class TextEmbedding(nn.Module):
  def __init__(self, vocab_size, embedding_dim, kernel_size, rnn_hidden_size):
    super(TextEmbedding, self).__init__()
    
    # Create the word embedding layer
    self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # Create the CNN layer
    self.cnn = nn.Conv1d(embedding_dim, embedding_dim, kernel_size, padding=kernel_size // 2)
    
    # Create the RNN layer
    self.rnn = nn.GRU(embedding_dim, rnn_hidden_size, batch_first=True)
    
  def forward(self, x):
    # Convert the input text to word embeddings
    x = self.word_embedding(x)
    
    # Extract local features with the CNN
    x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
    
    # Extract the global context with the RNN
    x, _ = self.rnn(x)
    
    # Take the mean of the RNN hidden states to create the text embedding
    text_embedding = x.mean(dim=1)
    
    return text_embedding


def tokenize_text(text):
  # Use the Punkt tokenizer to split the text into words and punctuation marks
  tokens = nltk.word_tokenize(text)

  return tokens

def cnvtensor(sentence):
    words = sentence.split()
    max_l = 0
    ts_list = []
    for w in words:
        ts_list.append(torch.ByteTensor(list(bytes(w, 'utf8'))))
        max_l = max(ts_list[-1].size()[0], max_l)

    w_t = torch.zeros((len(ts_list), max_l), dtype=torch.long)
    for i, ts in enumerate(ts_list):
        w_t[i, 0:ts.size()[0]] = ts

    return w_t

def preprocess_text(text):
  # Convert the text to a list of integers representing the words
    
    tokens = tokenize_text(text)
    tokens = list(text.lower().split())

    vocab, index = {}, 1  # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1
        vocab_size = len(vocab)

    inverse_vocab = {index: token for token, index in vocab.items()}


    tokens = [vocab[word] for word in tokens]

    token_size = len(tokens)
    # Convert the list of integers to a PyTorch tensor
    tokens = torch.tensor(tokens).unsqueeze(0)
    
    # Create the text embedding using the CNN-RNN model
    model = TextEmbedding(vocab_size, 256, 128, 1024)
    text_embedding = model(tokens)
    return text_embedding, token_size

if __name__ == "__main__":
    args =parse_args()
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.cfg_file != '':
        conf(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = DatasetLoadTransform(cfg.DATA_DIR,'train',
                              imsize=cfg.IMSIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GAN(output_dir)
        algo.train(dataloader, cfg.STAGE)

    else:
        text = "a kitchen counter with a rounded edge and shelves"
        text_embedding, token_size = preprocess_text(text)
        algo = GAN(output_dir)
        algo.sample(text_embedding, token_size, cfg.STAGE)
    

