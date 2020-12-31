import json

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)

    return list

rs = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/kimyaTerms_mixed.txt',"r").read()

# json.loads(string)
s = json.loads(rs)

keys = getList(s)

OXFORD_EN_GOLD_TERMS_DATASET_FILE_PATH = '/home/ikaraman/Desktop/oxfordDictionary/chemistry.txt'
TUBA_EN_GOLD_TERMS_DATASET_FILE_PATH = '/home/ikaraman/Desktop/tubaDictionary/Chemistry_en.txt'
termsGoldFileOxford=open(OXFORD_EN_GOLD_TERMS_DATASET_FILE_PATH,'r')
termsGoldFileTuba=open(TUBA_EN_GOLD_TERMS_DATASET_FILE_PATH,'r')
goldTerms = []
extractedTerms = []
for line in termsGoldFileOxford:
    if line.strip():
        goldTerms.append(line.lower().strip())
goldTerms=set(goldTerms)
matchCount = 0
for key in keys:
    extractedTerms.append(key.lower().strip())
extractedTerms=set(extractedTerms)
for term in extractedTerms:
    if term in goldTerms:
        matchCount = matchCount + 1
    else:
        print(term)

print("Precision for oxford: ", matchCount / len(extractedTerms))

goldTerms = []
extractedTerms = []
for line in termsGoldFileTuba:
    if line.strip():
        goldTerms.append(line.lower().strip())
goldTerms=set(goldTerms)
matchCount = 0
for key in keys:
    extractedTerms.append(key.lower().strip())
extractedTerms=set(extractedTerms)
for term in extractedTerms:
    if term in goldTerms:
        matchCount = matchCount + 1
    else:
        print(term)

#### bunda sebep tübanın çok kapsayıcı olmamasından kaynaklı olabilir.
print("Precision for tuba: ", matchCount / len(extractedTerms))

###### TODO SADECE TUBADA BULUNANLAR İÇİN ÇEVİRİ PRECISION'I HESAPLA