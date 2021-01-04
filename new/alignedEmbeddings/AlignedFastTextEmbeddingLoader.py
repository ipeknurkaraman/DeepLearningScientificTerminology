import numpy as np


class AlignedFastTextEmbeddingLoader:
    def loadEmbeddings(self, embeddingFilePath):
        file = open(embeddingFilePath)
        vocab_and_vectors = {}
        # put words as dict indexes and vectors as words values
        count=0
        for line in file:
            count+=1
            if count==1:
                continue
            values = line.split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
                vocab_and_vectors[word] = vector
            except:
                print('Error in vector.')
        return vocab_and_vectors

    ## tokenizer hem ingilizce hem türkçe kelimeleri içermeli
    def getEmbeddingMatrix(self, EN_embeddingFilePath, TR_embeddingFilePath, t):
        EN_vocab_and_vectors = self.loadEmbeddings(EN_embeddingFilePath)
        TR_vocab_and_vectors = self.loadEmbeddings(TR_embeddingFilePath)
        word_index = t.word_index
        vocab_size = len(word_index) + 1
        embed_size = 300
        sd = 1 / np.sqrt(embed_size)  # Standard deviation to use
        embedding_matrix = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
        # embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = EN_vocab_and_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_vector = TR_vocab_and_vectors.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix
