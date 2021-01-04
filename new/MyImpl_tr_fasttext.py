import json

import numpy
import string
from keras.preprocessing.text import Tokenizer
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
from evaluation.Evaluater import Evaluater
from tensorflow import keras
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from new.fasttext.FastTextEmbeddingLoader import FastTextEmbeddingLoader
from keras.models import load_model

# GOLD_TERMS_DATASET_FILE_PATH='/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt'
# SENTENCES_DATASET='/archive/EnglishSentencesDataset/Biology.txt'
# FIELD_NAME='biology'

GOLD_TERMS_DATASET_FILE_PATH='/home/ikaraman/Desktop/tubaDictionary/ComputerScience_tr.txt'
SENTENCES_DATASET='/archive/EnglishSentencesDataset/ComputerScience_tr.txt'
FIELD_NAME='computer'

MAX_SENTENCE_COUNT=10000
EPOCH=5

################# DEFINE TAGS #####################
BILOU_TAGS = ["B", "I", "L", "O", "U"]
n_tags = len(BILOU_TAGS)
tag2idx = {t: i for i, t in enumerate(BILOU_TAGS)}

################# RANDOM SEED #####################
# fix random seed for reproducibility
numpy.random.seed(7)

############### READ TERMS FROM DATASET #################
dataset1GramTerms = []
dataset2GramTerms = []
dataset3GramTerms = []
dataset4GramTerms = []
termsDatasetFile = open(GOLD_TERMS_DATASET_FILE_PATH, 'r')
goldTermCount=0
for term in termsDatasetFile.readlines():
    # if goldTermCount==2000:
    #     break
    if term.strip():
        term = term.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        if len(term.strip()) < 3:
            continue
        goldTermCount=goldTermCount+1
        ## TODO replace numbers like (1849-1936)
        wordCount = len(term.split())
        if wordCount is 1:
            dataset1GramTerms.append(term)
        if wordCount is 2:
            dataset2GramTerms.append(term)
        if wordCount is 3:
            dataset3GramTerms.append(term)
        if wordCount is 4:
            dataset4GramTerms.append(term)

# ################ READ SENTENCES ####################
# file = open('/home/ikaraman/Desktop/myfile.txt', 'r')
file = open(SENTENCES_DATASET, 'r')
sentences = file.readlines()

################### TAGGING #############################
sentencesAndTags = []
sentenceCount = 0
for sentence in sentences:
    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(sentences)))
    if sentenceCount == MAX_SENTENCE_COUNT:
        break
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    ######## if one word in the sentence, discard
    words = sentence.split()
    if len(words) == 1:
        continue
    #### Add <SOS> to begining, Add <EOS> to end
    sentence = " <SOS> " + sentence + " <EOS>"
    # +2 for SOS and EOS
    # initialize BILOU tags as all non terms
    tags = ['O' for i in range(len(words) + 2)]

    copySentence = str(sentence)
    for term4gram in dataset4GramTerms:
        term4gram = " " + str(term4gram) + str(" ")
        if term4gram in copySentence:
            copySentence = copySentence.replace(term4gram, " <B> <I> <I> <L> ")
    for term3gram in dataset3GramTerms:
        term3gram = " " + str(term3gram) + str(" ")
        if term3gram in copySentence:
            copySentence = copySentence.replace(term3gram, " <B> <I> <L> ")
    for term2gram in dataset2GramTerms:
        term2gram = " " + str(term2gram) + str(" ")
        if term2gram in copySentence:
            copySentence = copySentence.replace(term2gram, " <B> <L> ")
    for term1gram in dataset1GramTerms:
        term1gram = " " + str(term1gram) + str(" ")
        if term1gram in copySentence:
            copySentence = copySentence.replace(term1gram, " <U> ")

    # +2 for SOS and EOS
    # initialize BILOU tags as all non terms
    tags = ['O' for i in range(len(words) + 2)]
    i = 0
    for word in copySentence.split():
        if word in ["<B>", "<I>", "<L>", "<U>"]:
            tags[i] = word.replace("<", "").replace(">", "")
        else:
            print(word)
        i = i + 1
    sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1

print("TAGGING FINISHED.")

# split data
numpy.random.shuffle(sentencesAndTags)
trainingSet, validationSet, testSet = numpy.split(sentencesAndTags,
                                                  [int(len(sentencesAndTags) * 0.8), int(len(sentencesAndTags) * 0.9)])

trainingSet = numpy.concatenate([trainingSet, validationSet])
print("Training set:", len(trainingSet))
print("Test set:", len(testSet))

# find max sentence length
maxLength = 0
for sentencesAndTags in trainingSet:
    wordCount = len(str(sentencesAndTags[0]).split())
    if wordCount > maxLength:
        maxLength = wordCount
if maxLength>250:
    maxLength=250

########################### TRAININING PREPARATION ############################
# encode sentences
trainingDocs = [i[0] for i in trainingSet]
# prepare tokenizer
# encode unknown words as 1
t = Tokenizer(oov_token=1)
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
padded_tags = pad_sequences(maxlen=maxLength, sequences=trainingTags, padding="post", value=3)

padded_tags = [to_categorical(i, num_classes=n_tags) for i in padded_tags]

###################### EMBEDDING ##############################
embeddingLoader = FastTextEmbeddingLoader()
embedding_matrix = embeddingLoader.getEmbeddingMatrix('/archive/FastTextTurkishVectors/cc.tr.300.vec', t)
print("Embedding matrix created.")

# embeddingLoader = Test()
# embeddingLoader.getEmbeddingMatrix('/archive/trmodel',t)

############### MODEL CREATION #############################

# define model
model = Sequential()
# add embedding layer to model
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxLength, trainable=False))
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
bidirectionalLSTMLayer = LSTM(units=200, return_sequences=True, recurrent_dropout=0.1)  # variational biLSTM
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer2=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
outputLayer = TimeDistributed(Dense(n_tags, activation="softmax"))
#
model.add(bidirectionalLSTMLayer)
# model.add(bidirectionalLSTMLayer2)
model.add(outputLayer)

evaluator = Evaluater()
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m])
from seqeval.callbacks import F1Metrics

id2label = {0: 'B', 1: 'I', 2: 'L', 3: 'O', 4: 'U'}
callbacks = [F1Metrics(id2label)]

### TODO EPOCH
history = model.fit(padded_docs, numpy.array(padded_tags), batch_size=20, epochs=EPOCH, validation_split=0.1, verbose=1,
                    callbacks=callbacks)
print("Model created.")

# #### save model
# model.save('/archive/LSTMModels/modelByBiology.h5')  # creates a HDF5 file 'my_model.h5'
# #
# ### save tokenizer
# import pickle
# # saving
# with open('/archive/LSTMModels/biologyTokenizer.pickle', 'wb') as handle:
#     pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# tokenizer_json = t.to_json()
# with open('/archive/LSTMModels/biologyTokenizer.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

########################### TEST PREPERATION ########################
# encode sentences
testDocs = [i[0] for i in testSet]
# integer encode the documents
test_encoded_docs = t.texts_to_sequences(testDocs)
print(test_encoded_docs)
# encode tags
testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in testSet]]

# pad documents to a max length of words
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
test_padded_tags = pad_sequences(maxlen=maxLength, sequences=testTags, padding="post", value=3)

####||||||||||||||| evaluate on TEST DATA ########################
target_test = test_padded_tags.tolist()
y_pred = model.predict_classes(test_padded_docs)
y_pred = y_pred.tolist()

flat_target_list = []
for sublist in target_test:
    for item in sublist:
        flat_target_list.append(item)

flat_prediction_list = []
for sublist in y_pred:
    for item in sublist:
        flat_prediction_list.append(item)

for i in range(0, len(flat_target_list)):
    flat_target_list[i] = BILOU_TAGS[flat_target_list[i]]
for i in range(0, len(flat_prediction_list)):
    flat_prediction_list[i] = BILOU_TAGS[flat_prediction_list[i]]
# using naive method to
# perform conversion
print(classification_report(flat_target_list, flat_prediction_list))

extractedTerms = []
i = 0
for pred in y_pred:
    testSentence = testDocs[i]
    testSentenceWords = testSentence.split()
    tagPredictions = y_pred[i]
    j = 0
    previousTag=""
    for tagPrediction in tagPredictions:
        term = ""
        if 'B' == BILOU_TAGS[tagPrediction]:
            extractedTerms.append(testSentenceWords[j])
            previousTag='B'
        if 'I' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B','I']:
                lastWord = extractedTerms[len(extractedTerms) - 1]
                extractedTerms[len(extractedTerms) - 1] = lastWord + " "+(testSentenceWords[j])
            else:
                extractedTerms.append(testSentenceWords[j])
            previousTag='I'
        if 'L' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B','I']:
                lastWord = extractedTerms[len(extractedTerms) - 1]
                extractedTerms[len(extractedTerms) - 1] = lastWord + " "+(testSentenceWords[j])
            else:
                extractedTerms.append(testSentenceWords[j])
            previousTag='L'
        if 'U' == BILOU_TAGS[tagPrediction]:
            extractedTerms.append(testSentenceWords[j])
            previousTag='U'
        j=j+1
    i=i+1

extractedTerms=set(extractedTerms)
print(set(extractedTerms))
outputFilePath = '/home/ikaraman/Desktop/ExtractedTermsWithLSTM/' + FIELD_NAME + 'Terms_tr.txt'
with open(outputFilePath, 'w') as outputFile:
    for term in extractedTerms:
        outputFile.write(term+'\n')

print("Written file path:",outputFilePath)
print("Extracted term count:", len(extractedTerms))
print("Training sentence count:", len(trainingSet))