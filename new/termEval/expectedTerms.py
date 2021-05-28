termsExtractedFile= open('/archive/FinalOutputs/Chemistry/chemistry_expected_termsonTestData', 'r');

extractedTerms = []

multiWordCount=0
# goldTerms = set(goldTerms)

for line in termsExtractedFile.readlines():
    if line.strip():
        term=line.lower().strip()
        if len(str(term).split())>1:
            multiWordCount=multiWordCount+1




print("Multiword term count", multiWordCount)

