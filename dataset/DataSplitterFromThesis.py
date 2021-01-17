import numpy

import json

class Thesis(object):
    def __init__(self, data):
        self.__dict__ = json.loads(data)
FIELD_NAME = "ComputerScience"
# FIELD_NAME = "Biology"

# EN_SENTENCES_DATASET = '/archive/EnglishSentencesDataset/ComputerScience.txt'
# TR_SENTENCES_DATASET = '/archive/EnglishSentencesDataset/ComputerScience_tr.txt'
# THESIS_DATASET = '/archive/thesisOutput/Biology_thesis.txt'
THESIS_DATASET = '/archive/thesisOutput/Computer_thesis.txt'

THESIS_file = open(THESIS_DATASET, 'r')

thesisList = THESIS_file.readlines()
numpy.random.shuffle(thesisList)

trainingSet, validationSet, testSet = numpy.split(thesisList,
                                                  [int(len(thesisList) * 0.8),
                                                   int(len(thesisList) * 0.9)])
trainingSet = numpy.concatenate([trainingSet, validationSet])

# fix random seed for reproducibility
numpy.random.seed(7)

####### training ###############33
en_trainingSentences = []
tr_trainingSentences=[]
for line in trainingSet:
    thesis = Thesis(line)
    english_sentences = thesis.englishSentences
    turkishSentences = thesis.turkishSentences
    numpy.random.shuffle(english_sentences)
    numpy.random.shuffle(turkishSentences)
    for sentence in english_sentences:
        words = sentence.split()
        if len(words) == 1:
              continue
        en_trainingSentences.append(sentence)
    for sentence in turkishSentences:
        words = sentence.split()
        if len(words) == 1:
            continue
        tr_trainingSentences.append(sentence)

####### test ###############33
en_testSentences = []
tr_testSentences=[]
for line in testSet:
    thesis = Thesis(line)
    english_sentences = thesis.englishSentences
    turkish_Sentences = thesis.turkishSentences
    numpy.random.shuffle(english_sentences)
    numpy.random.shuffle(turkish_Sentences)
    for sentence in english_sentences:
        words = sentence.split()
        if len(words) == 1:
            continue
        en_testSentences.append(sentence)
    for sentence in turkish_Sentences:
        words = sentence.split()
        if len(words) == 1:
            continue
        tr_testSentences.append(sentence)


numpy.random.shuffle(en_trainingSentences)
numpy.random.shuffle(tr_trainingSentences)

tr_training_outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_training_tr.txt'
with open(tr_training_outputFilePath, 'w') as outputFile:
    for sentence in tr_trainingSentences:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')
en_training_outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_training_en.txt'
with open(en_training_outputFilePath, 'w') as outputFile:
    for sentence in en_trainingSentences:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')

tr_test_outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_test_tr.txt'
with open(tr_test_outputFilePath, 'w') as outputFile:
    for sentence in tr_testSentences:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')
en_test_outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_test_en.txt'
with open(en_test_outputFilePath, 'w') as outputFile:
    for sentence in en_testSentences:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')


