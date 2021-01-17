import numpy

# FIELD_NAME = "ComputerScience"
FIELD_NAME = "Biology"
ENGLISH_SENTENCE_SIZE = 40000
TURKISH_SENTENCE_SIZE = 40000

# EN_SENTENCES_DATASET = '/archive/EnglishSentencesDataset/ComputerScience.txt'
# TR_SENTENCES_DATASET = '/archive/EnglishSentencesDataset/ComputerScience_tr.txt'
EN_SENTENCES_DATASET = '/archive/EnglishSentencesDataset/Biology.txt'
TR_SENTENCES_DATASET = '/archive/EnglishSentencesDataset/Biology_tr.txt'

TR_file = open(TR_SENTENCES_DATASET, 'r')
EN_file = open(EN_SENTENCES_DATASET, 'r')

TR_sentences = TR_file.readlines()
numpy.random.shuffle(TR_sentences)
EN_sentences = EN_file.readlines()
numpy.random.shuffle(EN_sentences)

# fix random seed for reproducibility
numpy.random.seed(7)

####### TR ###############33
sentenceCount = 0
sentences = []
for sentence in TR_sentences:
    if sentenceCount == TURKISH_SENTENCE_SIZE:
        break
    words = sentence.split()
    if len(words) == 1:
        continue
    sentences.append(sentence)
    sentenceCount = sentenceCount + 1

numpy.random.shuffle(sentences)
trainingSet, validationSet, testSet = numpy.split(sentences,
                                                  [int(len(sentences) * 0.8),
                                                   int(len(sentences) * 0.9)])
trainingSet = numpy.concatenate([trainingSet, validationSet])

outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_training_tr_40.txt'
with open(outputFilePath, 'w') as outputFile:
    for sentence in trainingSet:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')
outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_test_tr_40.txt'
with open(outputFilePath, 'w') as outputFile:
    for sentence in testSet:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')

########## EN #################3
sentenceCount = 0
sentences = []
for sentence in EN_sentences:
    if sentenceCount == ENGLISH_SENTENCE_SIZE:
        break
    words = sentence.split()
    if len(words) == 1:
        continue
    sentences.append(sentence)
    sentenceCount = sentenceCount + 1

numpy.random.shuffle(sentences)
trainingSet, validationSet, testSet = numpy.split(sentences,
                                                  [int(len(sentences) * 0.8),
                                                   int(len(sentences) * 0.9)])
trainingSet = numpy.concatenate([trainingSet, validationSet])

outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_training_en_40.txt'
with open(outputFilePath, 'w') as outputFile:
    for sentence in trainingSet:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')
outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_test_en_40.txt'
with open(outputFilePath, 'w') as outputFile:
    for sentence in testSet:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')
