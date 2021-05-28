import string
import numpy
TEST_SENTENCES_DATASET='/archive/partitionedDataset/FoodScience_test_tr_1.txt'
testFile = open(TEST_SENTENCES_DATASET, 'r')
testSentences = testFile.readlines()
numpy.random.shuffle(testSentences)
sentenceCount=0
tr_testSentences=[]
for sentence in testSentences:
    if (sentenceCount%1000)==900:
        print("Tagging Progress: ", str(sentenceCount) + "/" + str(len(testSentences)))
        # break
    if sentenceCount==4000:
        break
    ######## if one word in the sentence, discard
    words = sentence.split()
    if len(words) == 1:
        continue
    sentenceCount=sentenceCount+1
    tr_testSentences.append(sentence)

print(sentenceCount)

FIELD_NAME='FoodScience'

tr_test_outputFilePath = '/archive/partitionedDataset/' + FIELD_NAME + '_test_tr.txt'
with open(tr_test_outputFilePath, 'w') as outputFile:
    for sentence in tr_testSentences:
        sentence=sentence.rstrip("\n")
        sentence=sentence.replace("\n","")
        outputFile.write(sentence + '\n')