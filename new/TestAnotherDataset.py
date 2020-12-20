import json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import string
import pickle

MAX_SENTENCE_COUNT = 1000000000

################# DEFINE TAGS #####################
BILOU_TAGS = ["B", "I", "L", "O", "U"]
n_tags = len(BILOU_TAGS)
tag2idx = {t: i for i, t in enumerate(BILOU_TAGS)}

### get pretrained model
from keras_preprocessing.text import tokenizer_from_json

model = load_model('/archive/LSTMModels/modelByBiology.h5',compile=False)

### get tokenizer
# with open('/archive/LSTMModels/biologyTokenizer.json') as f:
#     data = json.load(f)
#     t = tokenizer_from_json(data)
# loading
with open('/archive/LSTMModels/biologyTokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)

######## read test set #####
file = open('/archive/EnglishSentencesDataset/ComputerScience.txt', 'r')
sentences = file.readlines()

sentenceCount = 0
testDocs = []
for sentence in sentences:
    if sentenceCount == MAX_SENTENCE_COUNT:
        break
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    ######## if one word in the sentence, discard
    words = sentence.split()
    if len(words) == 1:
        continue
    #### Add <SOS> to begining, Add <EOS> to end
    sentence = " <SOS> " + sentence + " <EOS>"
    sentenceCount = sentenceCount + 1
    testDocs.append(sentence)

# encode sentences
# integer encode the documents
test_encoded_docs = t.texts_to_sequences(testDocs)
print(test_encoded_docs)
#### TODO Tagleri işaretleyip de setleyebilirsin ama gerek yok precision hesabı için
# encode tags
# testTags = [[tag2idx[w] for w in s] for s in [x[1] for x in testSet]]

# pad documents to a max length of words
maxLength=138 ## TODO bu model eğitirken mi gelmeli
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxLength, padding='post')
# pad also to tag sequence
# test_padded_tags = pad_sequences(maxlen=maxLength, sequences=testTags, padding="post", value=3)

####||||||||||||||| evaluate on TEST DATA ########################
y_pred = model.predict_classes(test_padded_docs)
y_pred = y_pred.tolist()

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
            previousTag='I'
        if 'L' == BILOU_TAGS[tagPrediction]:
            if previousTag in ['B','I']:
                lastWord = extractedTerms[len(extractedTerms) - 1]
                extractedTerms[len(extractedTerms) - 1] = lastWord + " "+(testSentenceWords[j])
            previousTag='L'
        if 'U' == BILOU_TAGS[tagPrediction]:
            extractedTerms.append(testSentenceWords[j])
            previousTag='U'
        j=j+1
    i=i+1

extractedTerms=set(extractedTerms)
print(set(extractedTerms))
with open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/computerTerms.txt', 'w') as outputFile:
    for term in extractedTerms:
        outputFile.write(term+'\n')

print("Finished with term count:",len(extractedTerms))