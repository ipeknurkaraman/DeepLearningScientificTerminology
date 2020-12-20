from xml.dom import minidom
import re
import string
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow import keras
import numpy
import random
from evaluation.Evaluater import Evaluater
from glove.EmbeddingLoader import EmbeddingLoader

t = Tokenizer(oov_token=1)
t.fit_on_texts(["a b c"])
vocab_size = len(t.word_index) + 1
# integer encode the documents
training_encoded_docs = t.texts_to_sequences(["a b c","b d c e"])
index = t.word_index
