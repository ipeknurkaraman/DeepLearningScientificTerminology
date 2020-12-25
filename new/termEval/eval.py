# termsGoldFile = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
# termsExtractedFile = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/biologyTerms.txt', 'r')
# termsGoldFile=open("/home/ikaraman/Desktop/tubaDictionary/Chemistry_tr.txt",'r')
# termsExtractedFile = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/chemistryTerms_tr.txt', 'r')
termsGoldFile=open("/home/ikaraman/Desktop/tubaDictionary/Biology_tr.txt",'r')
termsExtractedFile = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/biologyTerms_tr.txt', 'r')
goldTerms = []
extractedTerms = []
for line in termsGoldFile:
    if line.strip():
        goldTerms.append(line.lower().strip())
goldTerms=set(goldTerms)
matchCount = 0
for line in termsExtractedFile.readlines():
    if line.strip():
        extractedTerms.append(line.lower().strip())
extractedTerms=set(extractedTerms)
for term in extractedTerms:
    if term in goldTerms:
        matchCount = matchCount + 1
    else:
        print(term)

print("Precision", matchCount / len(extractedTerms))
