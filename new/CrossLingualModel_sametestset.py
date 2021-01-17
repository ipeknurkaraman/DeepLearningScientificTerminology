import string

import numpy
from seqeval.callbacks import F1Metrics
import numpy
import string
from keras_contrib.layers import CRF
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
from tensorflow import keras
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.models import load_model
from new.alignedEmbeddings.AlignedFastTextEmbeddingLoader import AlignedFastTextEmbeddingLoader
from new.evaluation.Evaluater import Evaluater

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


EPOCH = 5
EN_GOLD_TERMS_DATASET_FILE_PATH = '/home/ikaraman/Desktop/tubaDictionary/ComputerScience.txt'
TR_GOLD_TERMS_DATASET_FILE_PATH = '/home/ikaraman/Desktop/tubaDictionary/ComputerScience_tr.txt'

FIELD_NAME = 'computer'
EN_TRAINING_SENTENCES_DATASET = '/archive/partitionedDataset/ComputerScience_training_en.txt'
TR_TRAINING_SENTENCES_DATASET = '/archive/partitionedDataset/ComputerScience_training_tr.txt'
EN_TEST_SENTENCES_DATASET = '/archive/partitionedDataset/ComputerScience_test_en.txt'
TR_TEST_SENTENCES_DATASET = '/archive/partitionedDataset/ComputerScience_test_tr.txt'

# ################ READ SENTENCES ####################
EN_Training_file = open(EN_TRAINING_SENTENCES_DATASET, 'r')
TR_Training_file = open(TR_TRAINING_SENTENCES_DATASET, 'r')
EN_Test_file = open(EN_TEST_SENTENCES_DATASET, 'r')
TR_Test_file = open(TR_TEST_SENTENCES_DATASET, 'r')
EN_training_sentences = EN_Training_file.readlines()
TR_training_sentences = TR_Training_file.readlines()
EN_test_sentences = EN_Test_file.readlines()
TR_test_sentences = TR_Test_file.readlines()

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

########  TODO TR TAGGING #############
tr_training_sentencesAndTags = []
sentenceCount = 0
for sentence in TR_training_sentences:
    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(TR_training_sentences)))
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
    tr_training_sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1
print("TRAINING TR TAGGING FINISHED.")

########################### EN TAGGING #################################
en_training_sentencesAndTags = []
sentenceCount = 0
for sentence in EN_training_sentences:
    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(EN_training_sentences)))
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
    en_training_sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1

print("TRAINING EN TAGGING FINISHED.")

############################ TEST SET TAGGING #################################################################################

tr_test_sentencesAndTags = []
sentenceCount = 0
for sentence in TR_test_sentences:
    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(TR_test_sentences)))
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
    tr_test_sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1
print("TEST TR TAGGING FINISHED.")

########################### EN TAGGING #################################
en_test_sentencesAndTags = []
sentenceCount = 0
for sentence in EN_test_sentences:
    print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(EN_test_sentences)))
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
    en_test_sentencesAndTags.append((sentence, tags))
    sentenceCount = sentenceCount + 1

print("TEST EN TAGGING FINISHED.")
#####################################################################

trainingSet = numpy.concatenate([tr_training_sentencesAndTags, en_training_sentencesAndTags])
testSet = numpy.concatenate([tr_test_sentencesAndTags, en_test_sentencesAndTags])
print("Training set:", len(trainingSet))
print("Test set:", len(testSet))
print()
####################################
# find max sentence length
maxLength = 0
for sentencesAndTags in trainingSet:
    wordCount = len(str(sentencesAndTags[0]).split())
    if wordCount > maxLength:
        maxLength = wordCount
if maxLength > 250:
    maxLength = 250

########################### TRAININING PREPARATION ############################
# encode sentences
trainingDocs = [i[0] for i in trainingSet]
# prepare tokenizer
# encode unknown words as 1
tokenizer = Tokenizer(oov_token=1)
tokenizer.fit_on_texts(trainingDocs)
vocab_size = len(tokenizer.word_index) + 1
# integer encode the documents
training_encoded_docs = tokenizer.texts_to_sequences(trainingDocs)
print(training_encoded_docs)
# encode tags
trainingTags = [[tag2idx[w] for w in s] for s in [x[1] for x in trainingSet]]

# pad documents to a max length of words
padded_docs = pad_sequences(training_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
padded_tags = pad_sequences(maxlen=maxLength, sequences=trainingTags, padding="post", value=3)

padded_tags = [to_categorical(i, num_classes=n_tags) for i in padded_tags]

###################### EMBEDDING ##############################
embeddingLoader = AlignedFastTextEmbeddingLoader()
embedding_matrix = embeddingLoader.getEmbeddingMatrix('/archive/alignedFastTextVectors/wiki.en.align.vec',
                                                      '/archive/alignedFastTextVectors/wiki.tr.align.vec', tokenizer)
print("Embedding matrix created.")

############### MODEL CREATION #############################

# define model
model = Sequential()
# add embedding layer to model
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxLength, trainable=False))
# bidirectionalLSTMLayer=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer = LSTM(units=200, return_sequences=True, recurrent_dropout=0.1)  # variational biLSTM
bidirectionalLSTMLayer = Bidirectional(
    LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
# bidirectionalLSTMLayer2=Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))  # variational biLSTM
outputLayer = TimeDistributed(Dense(n_tags, activation="tanh"))
crf=CRF(len(BILOU_TAGS))
#
model.add(bidirectionalLSTMLayer)
model.add(outputLayer)
model.add(crf)

evaluator = Evaluater()
###### TODO optimezers
opt = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=opt, loss="categorical_crossentropy",
#               metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m])
model.compile(optimizer="rmsprop", loss=crf.loss_function,
                metrics=['acc', evaluator.f1_m, evaluator.precision_m, evaluator.recall_m,crf.accuracy])

id2label = {0: 'B', 1: 'I', 2: 'L', 3: 'O', 4: 'U'}
callbacks = [F1Metrics(id2label)]

BATCH_SIZE=50
### TODO EPOCH
history = model.fit(padded_docs, numpy.array(padded_tags), batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
print("Model created.")
######################################EN MODEL TEST##########################################
########################### TEST PREPERATION ########################
# encode sentences
en_testDocs = [i[0] for i in en_test_sentencesAndTags]
# integer encode the documents
test_encoded_docs = tokenizer.texts_to_sequences(en_testDocs)
print(test_encoded_docs)
# encode tags
testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in en_test_sentencesAndTags]]

# pad documents to a max length of words
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
test_padded_tags = pad_sequences(maxlen=maxLength, sequences=testTags, padding="post", value=3)

####||||||||||||||| evaluate on TEST DATA ########################
en_target_test = test_padded_tags.tolist()
en_y_pred = model.predict_classes(test_padded_docs)
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

######################################TR MODEL TEST##########################################
########################### TEST PREPERATION ########################
# encode sentences
tr_testDocs = [i[0] for i in tr_test_sentencesAndTags]
# integer encode the documents
test_encoded_docs = tokenizer.texts_to_sequences(tr_testDocs)
print(test_encoded_docs)
# encode tags
testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in tr_test_sentencesAndTags]]

# pad documents to a max length of words
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
test_padded_tags = pad_sequences(maxlen=maxLength, sequences=testTags, padding="post", value=3)

####||||||||||||||| evaluate on TEST DATA ########################
tr_target_test = test_padded_tags.tolist()
### TODO ?????????
tr_y_pred = model.predict_classes(test_padded_docs)
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
all_terms=[]
for pred in tr_y_pred:
    tr_testSentence = tr_testDocs[i]
    tr_testSentenceWords = tr_testSentence.split()
    j = 0
    previousTag = ""
    tr_extractedTerms = []
    for tagPrediction in pred:
        term = ""
        if 'B' == BILOU_TAGS[tagPrediction]:
            tr_extractedTerms.append(tr_testSentenceWords[j])
            previousTag = 'B'
        if 'I' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B', 'I']:
                if len(tr_extractedTerms) == 0:
                    ## TODO ne yapmalı
                    continue
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
    for term in tr_extractedTerms:
        all_terms.append(term)
    i = i + 1


extractedTerms = set(all_terms)
print(set(extractedTerms))
outputFilePath = '/home/ikaraman/Desktop/ExtractedTermsWithLSTM/' + FIELD_NAME + 'Terms_Mix_tr.txt'
with open(outputFilePath, 'w') as outputFile:
    for term in extractedTerms:
        outputFile.write(term + '\n')

print("Written file path:",outputFilePath)
print("Extracted term count:", len(extractedTerms))
print("Training sentence count:", len(trainingSet))
print("Epoch:",EPOCH)
print("Extracted Term Count:",len(extractedTerms))
