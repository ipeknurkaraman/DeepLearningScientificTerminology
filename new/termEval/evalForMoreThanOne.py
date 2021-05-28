
# termsGoldFile_EN = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
termsGoldFile = open("/home/ikaraman/Desktop/tubaDictionary/Chemistry_tr.txt", 'r')

FIELD_NAME = "foodscience"


# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_tr.txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/foodscienceTerms_Mix_tr (2).txt', 'r');
# termsExtractedFile= open('/home/ikaraman/Downloads/FoodScienceTerms_mid_tr.txt', 'r');
termsExtractedFile1= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_20_trTerms_tr.txt', 'r');
termsExtractedFile2= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_25_trTerms_tr.txt', 'r');
termsExtractedFile3= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_30_trTerms_tr.txt', 'r');
termsExtractedFile4= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_35_trTerms_tr.txt', 'r');
termsExtractedFile5= open('/archive/FinalOutputs/Chemistry/mono/Chemistry_40_trTerms_tr.txt', 'r');

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
# terms3 = getTerms(termsExtractedFile3)
# terms4 = getTerms(termsExtractedFile4)
# terms5 = getTerms(termsExtractedFile5)


def printGoldTermMatches(terms):
    matchCount = 0
    multiWordCount = 0
    oneWordCount = 0
    for term in terms:
        term=term.lower().strip()
        if term.replace("_<tr>","") in goldTerms:
            matchCount = matchCount + 1
            if len(term.split())>1:
                multiWordCount=multiWordCount+1
            if len(term.split())==1:
                oneWordCount=oneWordCount+1
    print("Total extracted term count:",len(terms))
    print("Match Count:",matchCount)
    print("Match Multiword Count:",multiWordCount)
    print("Match Oneword Count:",oneWordCount)
    print("Precision", matchCount / len(terms))


printGoldTermMatches(terms1)
printGoldTermMatches(terms2)
# printGoldTermMatches(terms3)
# printGoldTermMatches(terms4)
# printGoldTermMatches(terms5)

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
# intersection(terms2,terms3)
# intersection(terms3,terms4)
# intersection(terms4,terms5)

def difference(terms1,terms2):
    count=0
    multiWordCount=0
    for term in terms2:
        if term in goldTerms:
            if term.lower().strip() not in terms1:
                count=count+1
                print(term)
                if len(term.split())>1:
                    multiWordCount=multiWordCount+1
    print(count)
    print(multiWordCount)

difference(terms1,terms2)