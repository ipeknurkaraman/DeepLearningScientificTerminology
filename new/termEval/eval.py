from nltk.util import ngrams
import sys

# termsGoldFile_EN = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
termsGoldFile = open("/home/ikaraman/Desktop/tubaDictionary/ComputerScience_tr.txt", 'r')

FIELD_NAME = "foodscience"

# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/foodscienceTerms_Mix_tr (2).txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_mid_tr.txt', 'r');
# termsExtractedFile= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_25_trTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/Cheminew/termEval/eval.py:24stry_20-5Terms_Mix_4000_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Dow# termsExtractedFile= open('/home/ikaraman/Downloads/Chemistry_40_trTerms_tr.txt', 'r');nloads/Chemistry_40_trTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScienceTerms_25_yeni_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScienceTe\textbf{rms_40_yeni_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScienceTerms_25_yeni_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/Electronic_25_trTerm/s_tr.txt', 'r');
# termsExtractedFile = open('/home/ikaraman/Downloads/ComputerScienceTerms_30_yeni_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScience_35_trTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScience_30_trTerms_tr.txt', 'r');
termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScience_20-45Terms_tr.txt', 'r');

goldTerms = []
extractedTerms = []

for line in termsGoldFile:
    if line.strip():
        goldTerms.append(line.lower().strip())

# goldTerms = set(goldTerms)
matchCount = 0

for line in termsExtractedFile.readlines():
    if line.strip():
        term = line.lower().strip()
        # if len(term.split())>1:
        extractedTerms.append(term)
extractedTerms = set(extractedTerms)
subset = []
subsetFailureCount = 0
totalFailure = 0
for term in extractedTerms:
    term = term.lower().strip()
    term = term.replace("_<tr>", "")
    if term in goldTerms:
        matchCount = matchCount + 1
    else:
        totalFailure = totalFailure + 1
        for i in goldTerms:
            tekenized=i.split()
            if len(tekenized)==2:
               if term in tekenized:
                  subset.append(i + "_" + term)
                  subsetFailureCount = subsetFailureCount + 1
                  break
            if len(tekenized)==3:
                if term in tekenized:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
                biagrams=[" ".join(i) for i in list(ngrams(tekenized, 2))]
                if term in biagrams:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
            if len(tekenized)==4:
                if term in tekenized:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
                biagrams=[" ".join(i) for i in list(ngrams(tekenized, 2))]
                if term in biagrams:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
                trigrams=[" ".join(i) for i in list(ngrams(tekenized, 3))]
                if term in trigrams:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
            if len(tekenized)==5:
                if term in tekenized:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
                biagrams=[" ".join(i) for i in list(ngrams(tekenized, 2))]
                if term in biagrams:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
                trigrams=[" ".join(i) for i in list(ngrams(tekenized, 3))]
                if term in trigrams:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break
                fourgrams=[" ".join(i) for i in list(ngrams(tekenized, 4))]
                if term in fourgrams:
                    subset.append(i + "_" + term)
                    subsetFailureCount = subsetFailureCount + 1
                    break

# for term in goldTerms:
#     term=term.lower().strip()
#     if term not in extractedTerms:
#         print(term)

print("Extracted term count", len(extractedTerms))
print("Gold term count", len(goldTerms))
print("Match count", matchCount)
print("Precision", matchCount / len(extractedTerms))

print(set(subset))

print(totalFailure)
print(subsetFailureCount)
print(subsetFailureCount / totalFailure)
