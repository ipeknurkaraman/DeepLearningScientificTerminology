




import json

# Opening JSON file
f = open('/home/ikaraman/Downloads/data.json',)

# returns JSON object as
# a dictionary
data = json.load(f)
# Closing file
f.close()






# termsGoldFile_EN = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
termsGoldFile = open("/home/ikaraman/Desktop/tubaDictionary/ComputerScience_tr.txt", 'r')

FIELD_NAME = "foodscience"


# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/foodscienceTerms_Mix_tr (2).txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_mid_tr.txt', 'r');
# termsExtractedFile= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_25_trTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/Chemistry_20-5Terms_Mix_4000_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/Chemistry_40_trTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScienceTerms_25_yeni_tr.txt', 'r');
termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScience_30_trTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/ComputerScience_20-10Terms_tr.txt', 'r');

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
        if len(term.split())>1:
            extractedTerms.append(term)
extractedTerms=set(extractedTerms)
subset=[]
for term in extractedTerms:
    term=term.lower().strip()
    term=term.replace("_<tr>","")
    if term in goldTerms:
        matchCount = matchCount + 1
    # else:
    #     for i in goldTerms:
    #         if " "+ i in term:
    #             subset.append(i+"_"+term)

# for term in goldTerms:
#     term=term.lower().strip()
#     if term not in extractedTerms:
#         print(term)

print("Extracted term count", len(extractedTerms))
print("Gold term count", len(goldTerms))
print("Match count", matchCount)
print("Precision", matchCount / len(extractedTerms))


print(set(subset))