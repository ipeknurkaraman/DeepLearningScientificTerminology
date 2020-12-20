from xml.dom import minidom
from numpy import array
import re
import string
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
# from tensorflow import keras
from evaluation.Evaluater import Evaluater
from glove.EmbeddingLoader import EmbeddingLoader
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy
import random

BILOU_TAGS=["B","I","L","O","U"]
n_tags=len(BILOU_TAGS)
tag2idx = {t: i for i, t in enumerate(BILOU_TAGS)}


# fix random seed for reproducibility
# numpy.random.seed(7)

######################## GENIA DATA ######################################################
xmldoc = minidom.parse('/home/ikaraman/Desktop/LSTM Study/GENIA_term_3.02/GENIAcorpus3.02.xml')
# xmldoc = minidom.parse('/home/ikaraman/Desktop/LSTM Study/GENIA_term_3.02/test.xml')
itemlist = xmldoc.getElementsByTagName('abstract')
allTerms=[]
sentencesAndTags=[]
totalSentenceCount=0
ignoreCount=0
count=0
for item in itemlist:
    # if count>2:
    #     break
    count=count+1
    sentences = minidom.parseString(item.toxml()).getElementsByTagName('sentence')
    for sentence in sentences:
        lastSetTagIndex=0;
        totalSentenceCount=totalSentenceCount+1
        # print("Sentence Count",totalSentenceCount)
        consTags = sentence.getElementsByTagName('cons')
        originalSentence = re.sub('<[^<]+>', " ", sentence.toxml()).replace('(','').replace(')','').replace("[","").replace("]","").strip()
        originalSentence=originalSentence.translate(str.maketrans('','',string.punctuation)).lower().strip()
        originalSentence=originalSentence.replace("  "," ").strip()
        # originalSentence=re.sub('<[^<]+>', "", sentence.toxml())
        # print(originalSentence)
        words=originalSentence.split()
        if len(words)==1:
            continue
        # originalSentence=" <SOS> "+originalSentence+" <EOS>";
        print(originalSentence)
        # initialize BILOU tags as all non terms
        tags=['O' for i in range(len(words))]
        # +2 for SOS and EOS
        # tags=['O' for i in range(len(words)+2)]
        for cons in consTags:
            if cons.parentNode!=sentence:
                continue
            term = re.sub('<[^<]+>', "", cons.toxml()).replace('(','').replace(')','').strip()
            term=term.translate(str.maketrans('','',string.punctuation)).lower().strip()
            termElements=term.split()
            if len(termElements)==1:
                allSentenceStartIndexes=[]
                # Start with this value.
                location = -1
                # Loop while true.
                while True:
                    # Advance location by 1.
                    print("Finding:",term)
                    location = originalSentence.find(str(" "+term), location + 1)
                    # Break if not found.
                    if location == -1: break
                    if location+len(term)+1==len(originalSentence):
                        allSentenceStartIndexes.append(location)
                    else:
                        if not originalSentence[location+len(term)+1].isalpha():
                         allSentenceStartIndexes.append(location)
                allWordIndexes=[]
                for sentenceStartIndex in allSentenceStartIndexes:
                    firstPartOfSentences=originalSentence[0:sentenceStartIndex]
                    index=len(firstPartOfSentences.split())
                    if index <= lastSetTagIndex:
                          continue
                    allWordIndexes.append(index)
                for wordIndex in allWordIndexes:
                    print("Tags size:",len(tags),"Word Index:",wordIndex)
                    tags[wordIndex]='U'
                    lastSetTagIndex=wordIndex+len(termElements)-1

            else:
               print(term)
               allSentenceStartIndexes=[]
               # Start with this value.
               location = -1
               # Loop while true.
               while True:
                   # Advance location by 1.
                    location = originalSentence.find(str(" "+term), location + 1)
                   # Break if not found.
                    if location == -1: break
                    allSentenceStartIndexes.append(location)
               allWordIndexes=[]
               for sentenceStartIndex in allSentenceStartIndexes:
                   firstPartOfSentences=originalSentence[0:sentenceStartIndex]
                   index=len(firstPartOfSentences.split())
                   if index <= lastSetTagIndex:
                       continue
                   allWordIndexes.append(index)
               for wordIndex in allWordIndexes:
                   tags[wordIndex]='B'
                   for i in range(len(termElements)-2):
                     tags[wordIndex+1+i]='I'

                   tags[wordIndex+len(termElements)-1]='L'
                   lastSetTagIndex=wordIndex+len(termElements)-1

            allTerms.append(term)
        sentencesAndTags.append((originalSentence,tags))

sentencesAndTagsPreprocessed=[]
for sentenceAndTag in sentencesAndTags:
    # remove punctuation and convert to lower case
    sentencesAndTagsPreprocessed.append((str(sentenceAndTag[0].translate(str.maketrans('','',string.punctuation))).lower().strip(),sentenceAndTag[1]))
print(sentencesAndTagsPreprocessed)

# split data
numpy.random.shuffle(sentencesAndTagsPreprocessed)
trainingSet, validationSet, testSet = numpy.split(sentencesAndTagsPreprocessed, [int(len(sentencesAndTagsPreprocessed)*0.8), int(len(sentencesAndTagsPreprocessed)*0.9)])

trainingSet=numpy.concatenate([trainingSet,validationSet])
print("Training set:",len(trainingSet))
print("Test set:",len(testSet))

# find max sentence length
maxLength=0
for sentenceAndTagPreprocessed in trainingSet:
    wordCount = len(str(sentenceAndTagPreprocessed[0]).split())
    if wordCount>maxLength:
        maxLength=wordCount

########################### TRAININING PREPARATION ############################
# encode sentences
trainingDocs = [i[0] for i in trainingSet]
# prepare tokenizer
# encode unknown words as 1
t = Tokenizer(oov_token=1)
# t = Tokenizer()
t.fit_on_texts(trainingDocs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
training_encoded_docs = t.texts_to_sequences(trainingDocs)
print(training_encoded_docs)
# encode tags
trainingTags = [[tag2idx[w] for w in s] for s in [x[1] for x in trainingSet]]

# pad documents to a max length of words
padded_docs = pad_sequences(training_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
padded_tags= pad_sequences(maxlen=maxLength, sequences=trainingTags, padding="post", value=3)

padded_tags = [to_categorical(i, num_classes=n_tags) for i in padded_tags]

###################### EMBEDDING ##############################
embeddingLoader=EmbeddingLoader()
embedding_matrix = embeddingLoader.getEmbeddingMatrix('/archive/glove.6B.300d.txt',t)
print("Embedding matrix created.")

############### MODEL CREATION #############################
# define model
model = Sequential()
# add embedding layer to model
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxLength, trainable=False))
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
bidirectionalLSTMLayer=LSTM(units=200, return_sequences=True, recurrent_dropout=0.1)  # variational biLSTM
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer2=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
outputLayer=TimeDistributed(Dense(n_tags, activation="softmax"))
#
model.add(bidirectionalLSTMLayer)
# model.add(bidirectionalLSTMLayer2)
model.add(outputLayer)

evaluator=Evaluater()
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc',evaluator.f1_m,evaluator.precision_m, evaluator.recall_m])
from seqeval.callbacks import F1Metrics

id2label = {0: 'B', 1: 'I', 2: 'L',3:'O',4:'U'}
callbacks = [F1Metrics(id2label)]

history = model.fit(padded_docs, numpy.array(padded_tags), batch_size=5, epochs=5, validation_split=0.1, verbose=1,callbacks=callbacks)
print("Model created.")

########################### TEST PREPARTION ########################
# encode sentences
testDocs = [i[0] for i in testSet]
# prepare tokenizer
# t = Tokenizer()
# t.fit_on_texts(testDocs)
# integer encode the documents
test_encoded_docs = t.texts_to_sequences(testDocs)
print(test_encoded_docs)
# encode tags
testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in testSet]]


# pad documents to a max length of words
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
test_padded_tags= pad_sequences(maxlen=maxLength, sequences=testTags, padding="post", value=3)



# Final evaluation of the model
# loss, accuracy, f1_score, precision, recall = model.evaluate(test_padded_docs, numpy.array(test_padded_tags), verbose=0,batch_size=5)
# print("F1 Score:",str(f1_score));
# print("Precision",str(precision))
# print("Recall",str(recall))

# from sklearn.metrics import classification_report
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
target_test=test_padded_tags.tolist()
y_pred = model.predict_classes(test_padded_docs)
y_pred=y_pred.tolist()

flat_target_list = []
for sublist in target_test:
    for item in sublist:
        flat_target_list.append(item);

flat_prediction_list = []
for sublist in y_pred:
    for item in sublist:
        flat_prediction_list.append(item);

for i in range(0,len(flat_target_list)):
    flat_target_list[i]=BILOU_TAGS[flat_target_list[i]]
for i in range(0,len(flat_prediction_list)):
    flat_prediction_list[i]=BILOU_TAGS[flat_prediction_list[i]]
# using naive method to
# perform conversion
print(classification_report(flat_target_list, flat_prediction_list))


