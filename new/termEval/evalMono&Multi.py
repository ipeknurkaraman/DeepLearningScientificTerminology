
# termsGoldFile_EN = open('/home/ikaraman/Desktop/oxfordDictionary/biologyOxford.txt', 'r')
termsGoldFile = open("/home/ikaraman/Desktop/tubaDictionary/Electronic_tr.txt", 'r')

FIELD_NAME = "foodscience"


monoFile= open('/home/ikaraman/Downloads/Electronic_20_trTerms_tr.txt', 'r');
multiFile= open('/home/ikaraman/Downloads/Electronic_20-5Terms_tr.txt', 'r');

# monoFile= open('/archive/FinalOutputs/FoodScience/mono/FoodScienceTerms_30_tr.txt', 'r');
# multiFile= open('/archive/FinalOutputs/FoodScience/multi/FoodScience_20-10Terms_tr.txt', 'r');


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


monoTerms = getTerms(monoFile)
multiTerms= getTerms(multiFile)

def printGoldTermMatches(terms):
    matchCount = 0
    totalMultiWord = 0
    matchMultiWordCount = 0
    matchOneWordCount = 0
    totalOneWord=0
    for term in terms:
        if len(term.split())>1:
            totalMultiWord=totalMultiWord+1
        else:
            totalOneWord=totalOneWord+1
        term=term.lower().strip()
        if term.replace("_<tr>","") in goldTerms:
            matchCount = matchCount + 1
            if len(term.split())>1:
                matchMultiWordCount=matchMultiWordCount+1
            else:
                matchOneWordCount=matchOneWordCount+1
        # else:
        #     if len(term.split())>1:
        #        print(term)
    print("Total extracted term count:",len(terms))
    print("Total MultiWord Count:",totalMultiWord)
    print("Total OneWord Count:",totalOneWord)
    print("Match Count:",matchCount)
    print("Match MultiWord Count:",matchMultiWordCount)
    print("Match OneWord Count:",matchOneWordCount)
    print("Precision", matchCount / len(terms))


printGoldTermMatches(monoTerms)
printGoldTermMatches(multiTerms)

def intersection(mono,multi):
    intersectionCount = 0
    matcthIntersectionCount = 0
    for term in multi:
        term=term.replace("_<tr>","").lower().strip()
        if term in mono:
            intersectionCount = intersectionCount + 1
            if term in goldTerms:
                matcthIntersectionCount=matcthIntersectionCount+1

    print("Total Intersection Count:",intersectionCount)
    print("Total Match Intersection Count:",matcthIntersectionCount)

intersection(monoTerms,multiTerms)

mononunHatalıBuldukları=[]
# mononun bulduğu multinin bulamadığı kelimeler neler:
def differenceMonoVsMulti(mono,multi):
    count=0
    multiWordCount=0
    for term in mono:
        words=term.split(" ")
        term=""
        for word in words:
           term=term+" "+word+"_<tr>"
        if term.lower().strip() not in multi:
            # print(term)
            if term.replace("_<tr>","").lower().strip() in goldTerms:
                count=count+1
                # print(term)
                if len(term.split())>1:
                    multiWordCount=multiWordCount+1
            else:
                mononunHatalıBuldukları.append(term)
                print(term)
    # print(count)
    # print(multiWordCount)



# multinin bulduğu mononun bulamadığı kelimeler neler:
def differenceMultiVsMono(multi,mono):
    multiHata = 0
    count=0
    multiWordCount=0
    for term in multi:
        term=term.replace("_<tr>","").lower().strip()
        if term not in mono:
            # print(term)
            if term in goldTerms:
               count=count+1
               print(term)
               if len(term.split())>1:
                   multiWordCount=multiWordCount+1
            else:
                multiHata = multiHata+1
               # print(term)

    print(count)
    print(multiWordCount)
    return multiHata

# differenceMonoVsMulti(monoTerms,multiTerms)
multiHata = differenceMultiVsMono(multiTerms, monoTerms)

count=0
for term in mononunHatalıBuldukları:
    term = term.strip().replace("_<tr>","")
    for a in multiTerms:
        a = a.strip().replace("_<tr>","")
        if term==a:
          count=count+1

print("Mono hata:",len(mononunHatalıBuldukları))
print("Multi hata: ",multiHata)
print("Hata match",count)