#deal with tensors
import torch

#handling text data
from torchtext import data

#Reproducing same results
SEED = 2019

#Torch
torch.manual_seed(SEED)

#Cuda algorithms
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)


fields = [(None, None), ('text',TEXT),('label', LABEL)]

#loading custom dataset
training_data=data.TabularDataset(path = '/home/ikaraman/Desktop/quora.csv',format = 'csv',fields = fields,skip_header = True)


import random
train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))

#print preprocessed text
print(vars(training_data.examples[0]))

#initialize glove embeddings
TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")
LABEL.build_vocab(train_data)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
print(TEXT.vocab.freqs.most_common(10))

#Word dictionary
print(TEXT.vocab.stoi)

#check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)

print()
