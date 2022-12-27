import torch
import torch.nn as nn
import nltk
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
from configuration import cfg
# import stackgan_runner
# from stackgan_runner import GAN

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

model = torch.load('stackgan_model22.pth',map_location=torch.device('cpu'))
model.eval()



# Pre-process the input text
text = "church"
text_embedding, token_size = preprocess_text(text)

print(text_embedding)
nz = cfg.Z_DIM
# noise = Variable(torch.FloatTensor(batch_size, nz))
noise = torch.randn(cfg.TRAIN.BATCH_SIZE, 10)
noise.data.normal_(0, 1)
noise = noise.view(1,-1)
# Generate the image
for i in range(100):
  with torch.no_grad():
    image = model(text_embedding, noise)
    i = i + 1
# Post-process the image and save it
image = image[0]
image = image.squeeze(0)
image = transforms.ToPILImage()(image)
image.save('face.jpg')


