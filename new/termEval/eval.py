
# termsGoldFile_EN = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
termsGoldFile = open("/home/ikaraman/Desktop/tubaDictionary/FoodScience_tr.txt", 'r')

FIELD_NAME = "foodscience"


# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/foodscienceTerms_Mix_tr (2).txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_mid_tr.txt', 'r');
termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_Mix_Mid_tr.txt', 'r');

goldTerms = []
extractedTerms = []

for line in termsGoldFile:
    if line.strip():
        goldTerms.append(line.lower().strip())

# goldTerms = set(goldTerms)
matchCount = 0

for line in termsExtractedFile.readlines():
    if line.strip():
        term=line.lower().strip()
        extractedTerms.append(term)

for term in extractedTerms:
    term=term.lower().strip()
    if term in goldTerms:
        matchCount = matchCount + 1

for term in goldTerms:
    term=term.lower().strip()
    if term not in extractedTerms:
        print(term)

print("Extracted term count", len(extractedTerms))
print("Gold term count", len(goldTerms))
print("Match count", matchCount)
print("Precision", matchCount / len(extractedTerms))

