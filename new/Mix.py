import string

import numpy
import numpy
import string
from keras_contrib.layers import CRF
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from glove.EmbeddingLoader import EmbeddingLoader
from fasttext.FastTextEmbeddingLoader import FastTextEmbeddingLoader
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
from keras.models import load_model

MAX_SENTENCE_COUNT = 50000
EPOCH = 5
fieldName = 'kimya'
SENTENCES_DATASET = '/archive/TERM_EXTRACTION_DATASET/COMMON/' + fieldName + '/SentenceAlignments'

sentencesDatasetFile = open(SENTENCES_DATASET, 'r')
enSentences = []
trSentences = []
for line in sentencesDatasetFile.readlines():
    split = line.split("|||")
    if len(split) == 2:
        enSentences.append(split[0].strip())
        trSentences.append(split[1].strip())

TR_GOLD_TERMS_DATASET_FILE_PATH = '/home/ikaraman/Desktop/tubaDictionary/Chemistry_tr.txt'
EN_GOLD_TERMS_DATASET_FILE_PATH = '/home/ikaraman/Desktop/oxfordDictionary/chemistry.txt'

################# DEFINE TAGS #####################
BILOU_TAGS = ["B", "I", "L", "O", "U"]
n_tags = len(BILOU_TAGS)
tag2idx = {t: i for i, t in enumerate(BILOU_TAGS)}

################# RANDOM SEED #####################
# fix random seed for reproducibility
numpy.random.seed(7)

############### READ TR GOLD_TERMS FROM DATASET #################
tr_dataset1GramTerms = []
tr_dataset2GramTerms = []
tr_dataset3GramTerms = []
tr_dataset4GramTerms = []
termsDatasetFile = open(TR_GOLD_TERMS_DATASET_FILE_PATH, 'r')
goldTermCount = 0
for term in termsDatasetFile.readlines():
    # if goldTermCount==2000:
    #     break
    if term.strip():
        term = term.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        if len(term.strip()) < 3:
            continue
        goldTermCount = goldTermCount + 1
        ## TODO replace numbers like (1849-1936)
        wordCount = len(term.split())
        if wordCount is 1:
            tr_dataset1GramTerms.append(term)
        if wordCount is 2:
            tr_dataset2GramTerms.append(term)
        if wordCount is 3:
            tr_dataset3GramTerms.append(term)
        if wordCount is 4:
            tr_dataset4GramTerms.append(term)

############### READ EN GOLD_TERMS FROM DATASET #################
en_dataset1GramTerms = []
en_dataset2GramTerms = []
en_dataset3GramTerms = []
en_dataset4GramTerms = []
termsDatasetFile = open(EN_GOLD_TERMS_DATASET_FILE_PATH, 'r')
goldTermCount = 0
for term in termsDatasetFile.readlines():
    # if goldTermCount==2000:
    #     break
    if term.strip():
        term = term.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        if len(term.strip()) < 3:
            continue
        goldTermCount = goldTermCount + 1
        ## TODO replace numbers like (1849-1936)
        wordCount = len(term.split())
        if wordCount is 1:
            en_dataset1GramTerms.append(term)
        if wordCount is 2:
            en_dataset2GramTerms.append(term)
        if wordCount is 3:
            en_dataset3GramTerms.append(term)
        if wordCount is 4:
            en_dataset4GramTerms.append(term)

######## TR TAGGING #############
tr_sentencesAndTags = []
sentenceCount = 0
for sentence in trSentences:
    if sentenceCount == MAX_SENTENCE_COUNT:
        break
    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(trSentences)))
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
    for term4gram in tr_dataset4GramTerms:
        term4gram = " " + str(term4gram) + str(" ")
        if term4gram in copySentence:
            copySentence = copySentence.replace(term4gram, " <B> <I> <I> <L> ")
    for term3gram in tr_dataset3GramTerms:
        term3gram = " " + str(term3gram) + str(" ")
        if term3gram in copySentence:
            copySentence = copySentence.replace(term3gram, " <B> <I> <L> ")
    for term2gram in tr_dataset2GramTerms:
        term2gram = " " + str(term2gram) + str(" ")
        if term2gram in copySentence:
            copySentence = copySentence.replace(term2gram, " <B> <L> ")
    for term1gram in tr_dataset1GramTerms:
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
    tr_sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1
print("TR TAGGING FINISHED.")

########################### EN TAGGING #################################
en_sentencesAndTags = []
sentenceCount = 0
for sentence in enSentences:

    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(enSentences)))
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
    for term4gram in en_dataset4GramTerms:
        term4gram = " " + str(term4gram) + str(" ")
        if term4gram in copySentence:
            copySentence = copySentence.replace(term4gram, " <B> <I> <I> <L> ")
    for term3gram in en_dataset3GramTerms:
        term3gram = " " + str(term3gram) + str(" ")
        if term3gram in copySentence:
            copySentence = copySentence.replace(term3gram, " <B> <I> <L> ")
    for term2gram in en_dataset2GramTerms:
        term2gram = " " + str(term2gram) + str(" ")
        if term2gram in copySentence:
            copySentence = copySentence.replace(term2gram, " <B> <L> ")
    for term1gram in en_dataset1GramTerms:
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
    en_sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1

print("EN TAGGING FINISHED.")

#############################################################################################################

# merge en and tr sentences ---> list of (tuple,tuple) first tuple for english, second tuple for turkish
# tuple: (sentence,tags)
mergeSentencesAndTags = []
i = 0
for x in en_sentencesAndTags:
    y = tr_sentencesAndTags[i]
    mergeSentencesAndTags.append((x, y))
    i = i + 1
# split data
numpy.random.shuffle(mergeSentencesAndTags)
trainingSet, validationSet, testSet = numpy.split(mergeSentencesAndTags,
                                                  [int(len(mergeSentencesAndTags) * 0.8),
                                                   int(len(mergeSentencesAndTags) * 0.9)])

trainingSet = numpy.concatenate([trainingSet, validationSet])
print("Training set:", len(trainingSet))
print("Test set:", len(testSet))
print()

i = 0
en_training_set = []
for x in trainingSet:
    en_training_set.append(trainingSet[i][0])
    i = i + 1
i = 0
tr_training_set = []
for x in trainingSet:
    tr_training_set.append(trainingSet[i][1])
    i = i + 1
i = 0
en_test_set = []
for x in testSet:
    en_test_set.append(testSet[i][0])
    i = i + 1
i = 0
tr_test_set = []
for x in testSet:
    tr_test_set.append(testSet[i][1])
    i = i + 1
print()
################################## EN MODEL #######################################
# find max sentence length
en_maxLength = 0
for sentencesAndTags in trainingSet:
    wordCount = len(str(sentencesAndTags[0]).split())
    if wordCount > en_maxLength:
        en_maxLength = wordCount
if en_maxLength > 250:
    maxLength = 250

########################### TRAININING PREPARATION ############################
# encode sentences
trainingDocs = [i[0] for i in en_training_set]
# prepare tokenizer
# encode unknown words as 1
en_tokinezer = Tokenizer(oov_token=1)
en_tokinezer.fit_on_texts(trainingDocs)
vocab_size = len(en_tokinezer.word_index) + 1
# integer encode the documents
training_encoded_docs = en_tokinezer.texts_to_sequences(trainingDocs)
print(training_encoded_docs)
# encode tags
trainingTags = [[tag2idx[w] for w in s] for s in [x[1] for x in en_training_set]]

# pad documents to a max length of words
padded_docs = pad_sequences(training_encoded_docs, maxlen=en_maxLength, padding='post')
# pad also to tag sequence
padded_tags = pad_sequences(maxlen=en_maxLength, sequences=trainingTags, padding="post", value=3)

padded_tags = [to_categorical(i, num_classes=n_tags) for i in padded_tags]

###################### EMBEDDING ##############################
en_embeddingLoader = EmbeddingLoader()
en_embedding_matrix = en_embeddingLoader.getEmbeddingMatrix('/archive/glove.6B.300d.txt', en_tokinezer)
print("EN Embedding matrix created.")

############### MODEL CREATION #############################

# define model
enModel = Sequential()
# add embedding layer to model
enModel.add(Embedding(vocab_size, 300, weights=[en_embedding_matrix], input_length=en_maxLength, trainable=False))
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer = LSTM(units=200, return_sequences=True, recurrent_dropout=0.1)  # variational biLSTM
bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer2=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
outputLayer = TimeDistributed(Dense(n_tags, activation="tanh"))
crf=CRF(len(BILOU_TAGS))
#
enModel.add(bidirectionalLSTMLayer)
enModel.add(outputLayer)
enModel.add(crf)

evaluator = Evaluater()
###### TODO optimezers
opt = keras.optimizers.Adam(learning_rate=0.001)
# enModel.compile(optimizer=opt, loss="categorical_crossentropy",
#                 metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m,crf.accuracy])
enModel.compile(optimizer="rmsprop", loss=crf.loss_function,
                metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m,crf.accuracy])
from seqeval.callbacks import F1Metrics

id2label = {0: 'B', 1: 'I', 2: 'L', 3: 'O', 4: 'U'}
callbacks = [F1Metrics(id2label)]

### TODO EPOCH
history = enModel.fit(padded_docs, numpy.array(padded_tags), batch_size=20, epochs=EPOCH, validation_split=0.1,
                      verbose=1,
                      callbacks=callbacks)
print("EN Model created.")
######################################EN MODEL TEST##########################################
########################### TEST PREPERATION ########################
# encode sentences
en_testDocs = [i[0] for i in en_test_set]
# integer encode the documents
test_encoded_docs = en_tokinezer.texts_to_sequences(en_testDocs)
print(test_encoded_docs)
# encode tags
testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in en_test_set]]

# pad documents to a max length of words
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=en_maxLength, padding='post')
# pad also to tag sequence
test_padded_tags = pad_sequences(maxlen=en_maxLength, sequences=testTags, padding="post", value=3)

####||||||||||||||| evaluate on TEST DATA ########################
en_target_test = test_padded_tags.tolist()
en_y_pred = enModel.predict_classes(test_padded_docs)
en_y_pred = en_y_pred.tolist()

flat_target_list = []
for sublist in en_target_test:
    for item in sublist:
        flat_target_list.append(item)

flat_prediction_list = []
for sublist in en_y_pred:
    for item in sublist:
        flat_prediction_list.append(item)

for i in range(0, len(flat_target_list)):
    flat_target_list[i] = BILOU_TAGS[flat_target_list[i]]
for i in range(0, len(flat_prediction_list)):
    flat_prediction_list[i] = BILOU_TAGS[flat_prediction_list[i]]
# using naive method to
# perform conversion
print(classification_report(flat_target_list, flat_prediction_list))

############################################## TR MODEL #################
########################### TRAININING PREPARATION ############################
# encode sentences
trainingDocs = [i[0] for i in tr_training_set]
# prepare tokenizer
# encode unknown words as 1
tr_tokenizer = Tokenizer(oov_token=1)
tr_tokenizer.fit_on_texts(trainingDocs)
vocab_size = len(tr_tokenizer.word_index) + 1
# integer encode the documents
training_encoded_docs = tr_tokenizer.texts_to_sequences(trainingDocs)
print(training_encoded_docs)
# encode tags
trainingTags = [[tag2idx[w] for w in s] for s in [x[1] for x in tr_training_set]]

# pad documents to a max length of words
padded_docs = pad_sequences(training_encoded_docs, maxlen=en_maxLength, padding='post')
# pad also to tag sequence
padded_tags = pad_sequences(maxlen=en_maxLength, sequences=trainingTags, padding="post", value=3)

padded_tags = [to_categorical(i, num_classes=n_tags) for i in padded_tags]

###################### EMBEDDING ##############################
tr_embeddingLoader = FastTextEmbeddingLoader()
tr_embedding_matrix = tr_embeddingLoader.getEmbeddingMatrix('/archive/FastTextTurkishVectors/cc.tr.300.vec', tr_tokenizer)
print("TR Embedding matrix created.")

############### MODEL CREATION #############################

# define model
trModel = Sequential()
# add embedding layer to model
trModel.add(Embedding(vocab_size, 300, weights=[tr_embedding_matrix], input_length=en_maxLength, trainable=False))
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer = LSTM(units=200, return_sequences=True, recurrent_dropout=0.1)  # variational biLSTM
bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer2=Bidirectional(LST (units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
outputLayer = TimeDistributed(Dense(n_tags, activation="tanh"))
crf=CRF(len(BILOU_TAGS))
#
trModel.add(bidirectionalLSTMLayer)
# model.add(bidirectionalLSTMLayer2)
trModel.add(outputLayer)
trModel.add(crf)

evaluator = Evaluater()
####### TODO optimizers
opt = keras.optimizers.Adam(learning_rate=0.001)
# trModel.compile(optimizer=opt, loss="categorical_crossentropy",
#                 metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m,crf.accuracy])
trModel.compile(optimizer="rmsprop", loss=crf.loss_function,
                metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m,crf.accuracy])
from seqeval.callbacks import F1Metrics

id2label = {0: 'B', 1: 'I', 2: 'L', 3: 'O', 4: 'U'}
callbacks = [F1Metrics(id2label)]

### TODO EPOCH
history = trModel.fit(padded_docs, numpy.array(padded_tags), batch_size=20, epochs=EPOCH, validation_split=0.1,
                      verbose=1,
                      callbacks=callbacks)
print("TR Model created.")
######################################TR MODEL TEST##########################################
########################### TEST PREPERATION ########################
# encode sentences
tr_testDocs = [i[0] for i in tr_test_set]
# integer encode the documents
test_encoded_docs = tr_tokenizer.texts_to_sequences(tr_testDocs)
print(test_encoded_docs)
# encode tags
testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in tr_test_set]]

# pad documents to a max length of words
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=en_maxLength, padding='post')
# pad also to tag sequence
test_padded_tags = pad_sequences(maxlen=en_maxLength, sequences=testTags, padding="post", value=3)

####||||||||||||||| evaluate on TEST DATA ########################
tr_target_test = test_padded_tags.tolist()
tr_y_pred = trModel.predict_classes(test_padded_docs)
tr_y_pred = tr_y_pred.tolist()

flat_target_list = []
for sublist in tr_target_test:
    for item in sublist:
        flat_target_list.append(item)

flat_prediction_list = []
for sublist in tr_y_pred:
    for item in sublist:
        flat_prediction_list.append(item)

for i in range(0, len(flat_target_list)):
    flat_target_list[i] = BILOU_TAGS[flat_target_list[i]]
for i in range(0, len(flat_prediction_list)):
    flat_prediction_list[i] = BILOU_TAGS[flat_prediction_list[i]]
# using naive method to
# perform conversion
print(classification_report(flat_target_list, flat_prediction_list))

########################### EXTRACT TERMS ##################################3
i = 0
all_extracted_terms=[]
for pred in en_y_pred:
    en_testSentence = en_testDocs[i]
    tr_testSentence = tr_testDocs[i]
    en_testSentenceWords = en_testSentence.split()
    tr_testSentenceWords = tr_testSentence.split()
    en_tagPredictions = en_y_pred[i]
    tr_tagPredictions = tr_y_pred[i]
    j = 0
    previousTag = ""
    en_extractedTerms = []
    for tagPrediction in en_tagPredictions:
        term = ""
        if 'B' == BILOU_TAGS[tagPrediction]:
            en_extractedTerms.append(en_testSentenceWords[j])
            previousTag = 'B'
        if 'I' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B', 'I']:
                lastWord = en_extractedTerms[len(en_extractedTerms) - 1]
                en_extractedTerms[len(en_extractedTerms) - 1] = lastWord + " " + (en_testSentenceWords[j])
            previousTag = 'I'
        if 'L' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B', 'I']:
                lastWord = en_extractedTerms[len(en_extractedTerms) - 1]
                en_extractedTerms[len(en_extractedTerms) - 1] = lastWord + " " + (en_testSentenceWords[j])
            previousTag = 'L'
        if 'U' == BILOU_TAGS[tagPrediction]:
            en_extractedTerms.append(en_testSentenceWords[j])
            previousTag = 'U'
        j = j + 1

    j = 0
    previousTag = ""
    tr_extractedTerms = []
    for tagPrediction in tr_tagPredictions:
        term = ""
        if 'B' == BILOU_TAGS[tagPrediction]:
            tr_extractedTerms.append(tr_testSentenceWords[j])
            previousTag = 'B'
        if 'I' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B', 'I']:
                lastWord = tr_extractedTerms[len(tr_extractedTerms) - 1]
                tr_extractedTerms[len(tr_extractedTerms) - 1] = lastWord + " " + (tr_testSentenceWords[j])
            previousTag = 'I'
        if 'L' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B', 'I']:
                lastWord = tr_extractedTerms[len(tr_extractedTerms) - 1]
                tr_extractedTerms[len(tr_extractedTerms) - 1] = lastWord + " " + (tr_testSentenceWords[j])
            previousTag = 'L'
        if 'U' == BILOU_TAGS[tagPrediction]:
            tr_extractedTerms.append(tr_testSentenceWords[j])
            previousTag = 'U'
        j = j + 1
    i = i + 1
    all_extracted_terms.append([en_extractedTerms,tr_extractedTerms])

en_extractedTerms = set(en_extractedTerms)
print(set(en_extractedTerms))

from collections import Counter
termSet={}
for i in all_extracted_terms:
    enTerms=i[0]
    trTerms=i[1]
    if len(enTerms)==0:
        continue
    for enTerm in enTerms:
        for trTerm in trTerms:
            if enTerm not in termSet.keys():
                termSet[enTerm]=Counter()
            termSet[enTerm].update([trTerm])
print(termSet)

import json
with open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/'+fieldName+'Terms_mixed.txt', 'w',encoding='utf-8') as outputFile:
    outputFile.write(json.dumps(termSet, ensure_ascii=False))

print("Finished with term count:",len(termSet))