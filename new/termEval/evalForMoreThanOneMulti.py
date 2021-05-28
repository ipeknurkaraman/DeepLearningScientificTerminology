
# termsGoldFile_EN = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
termsGoldFile = open("/home/ikaraman/Desktop/tubaDictionary/Chemistry_tr.txt", 'r')

FIELD_NAME = "foodscience"


# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/foodscienceTerms_Mix_tr (2).txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_mid_tr.txt', 'r');
termsExtractedFile1= open('/archive/FinalOutputs/Chemistry/multi/Chemistry_20-5Terms_tr.txt', 'r');
termsExtractedFile2= open('/archive/FinalOutputs/Chemistry/multi/Chemistry_20-10Terms_tr.txt', 'r');
termsExtractedFile3= open('/archive/FinalOutputs/Chemistry/multi/Chemistry_20-15Terms_tr.txt', 'r');
termsExtractedFile4= open('/archive/FinalOutputs/Chemistry/multi/Chemistry_20-20Terms_tr.txt', 'r');

goldTerms = []
for line in termsGoldFile:
    if line.strip():
        goldTerms.append(line.lower().strip())



def getTerms(termExtractedFile):
    extractedTerms = []
    for line in termExtractedFile.readlines():
        if line.strip():
          term=line.lower().strip()
          extractedTerms.append(term)
    return extractedTerms


terms1 = getTerms(termsExtractedFile1)
terms2 = getTerms(termsExtractedFile2)
terms3 = getTerms(termsExtractedFile3)
terms4 = getTerms(termsExtractedFile4)


def printGoldTermMatches(terms):
    matchCount = 0
    for term in terms:
        term=term.lower().strip()
        if term.replace("_<tr>","") in goldTerms:
            matchCount = matchCount + 1
    print("Total extracted term count:",len(terms))
    print("Match Count:",matchCount)
    print("Precision", matchCount / len(terms))


printGoldTermMatches(terms1)
printGoldTermMatches(terms2)
printGoldTermMatches(terms3)
printGoldTermMatches(terms4)

def intersection(terms1,terms2):
    intersectionCount = 0
    matcthIntersectionCount = 0
    for term in terms2:
        term=term.lower().strip()
        if term in terms1:
            intersectionCount = intersectionCount + 1
            if term.replace("_<tr>","") in goldTerms:
                matcthIntersectionCount=matcthIntersectionCount+1

    print("Total Intersection Count:",intersectionCount)
    print("Total Match Intersection Count:",matcthIntersectionCount)

intersection(terms1,terms2)
intersection(terms2,terms3)
intersection(terms3,terms4)