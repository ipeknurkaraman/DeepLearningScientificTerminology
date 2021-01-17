FIELD_NAME="computer"
termsGoldFile=open("/home/ikaraman/Desktop/tubaDictionary/ComputerScience_tr.txt",'r')

extractedTerms = ['değişmez', 'büyük veri', 'adaya dönüş', 'ikili', 'mimari', 'birleştirme', 'yazılım mühendisliği',
             'bağlama', 'bilgi işleme', 'işletim sistemi', 'sözlük', 'sözcük', 'işlem', 'damga', 'anahtar', 'bildirim',
             'elde', 'bağlantı', 'işaret', 'geri yayılım', 'ağaç', 'öznitelik', 'fonksiyon', 'olay işleme',
             'bilgi toplama', 'bulut bilişim', 'önbelleği', 'hedef', 'veri kümesi', 'kullanıcı', 'veri işleme',
             'sunucu', 'akıllı telefon', 'bilgi', 'sıralama', 'çıkan kod', 'veri madenciliği', 'taşma', 'bileşen',
             'metin madenciliği', 'kimlik denetim', 'istemcisi', 'aktif', 'internete', 'feature', 'sanallaştırma',
             'vector', 'genetik algoritma', 'veri güvenliği', 'page', 'bulanık', 'sınıf', 'varlık', 'yazılım', 'hata',
             'yazmaç', 'duygu analizi', 'boşluk', 'saldırı', 'bağ', 'yakalama', 'gerçek zamanlı', 'dosya',
             'bilgi güvenliği', 'konum', 'sürüm', 'görev', 'ifade', 'network', 'kaldıraç', 'yazılım mimarisi',
             'konumlandırma', 'örüntü', 'veri', 'özellik', 'karakter', 'sanal', 'bağlam', 'tree', 'veri toplam',
             'veri ambarı', 'modül', 'duygu tanıma', 'sıra', 'önbellek', 'kodlama', 'ilgi', 'veritabanı'
    , 'veri doğrulama'
    , 'hizmet'
    , 'kök'
    , 'işaret işleme'
    , 'çevrimiçi'
    , 'application'
    , 'paket'
    , 'bilgi paketi'
    , 'listesi'
    , 'alan'
    , 'görselleştirme'
    , 'süreç'
    , 'şifreleme'
    , 'olay'
    , 'açık'
    , 'kaynak'
    , 'uzaktan eğitim'
    , 'kimlik doğrulama'
    , 'girdi'
    , 'yapay zeka'
    , 'big data'
    , 'uygulama'
    , 'evrim'
    , 'function'
    , 'bölütleme'
    , 'ağırlık'
    , 'tip'
    , 'sorgu'
    , 'parmak izi'
    , 'vektör'
    , 'belge'
    , 'trafik'
    , 'işaretçi'
    , 'multilayer perceptron'
    , 'günlük'
    , 'çıktısı'
    , 'erişim'
    , 'akıllı ev'
    , 'routing'
    , 'çıkan'
    , 'kod'
    , 'bilişim'
    , 'doğrulama'
    , 'konumsal'
    , 'bölümlendirme'
    , 'database'
    , 'konfigürasyon'
    , 'nesne'
    , 'information'
    , 'yük' \
      'internet']
# termsGoldFile=open("/home/ikaraman/Desktop/tubaDictionary/ComputerScience.txt",'r')
# termsExtractedFile = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/'+FIELD_NAME+'Terms.txt', 'r')
# FIELD_NAME="chemistry"
# termsGoldFile=open('/home/ikaraman/Desktop/oxfordDictionary/chemistry.txt','r')
# termsExtractedFile = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/'+FIELD_NAME+'Terms.txt', 'r')
# termsGoldFile=open('/home/ikaraman/Desktop/tubaDictionary/Chemistry_tr.txt','r')
# termsExtractedFile = open('/home/ikaraman/Desktop/ExtractedTermsWithLSTM/'+FIELD_NAME+'Terms_tr.txt', 'r')
goldTerms = []
for line in termsGoldFile:
    if line.strip():
        goldTerms.append(line.lower().strip())
goldTerms=set(goldTerms)
matchCount = 0

extractedTerms=set(extractedTerms)
for term in extractedTerms:
    if term in goldTerms:
        matchCount = matchCount + 1
    else:
        print(term)

print("Extracted term count",len(extractedTerms))
print("Gold term count",len(goldTerms))
print("Match count",matchCount)
print("Precision", matchCount / len(extractedTerms))
