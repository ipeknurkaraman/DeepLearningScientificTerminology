from numpy import asarray
from numpy import zeros
import numpy as np

class EmbeddingLoader:
    def loadEmbeddings(self,embeddingFilePath):
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(embeddingFilePath)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def getEmbeddingMatrix(self,embeddingFilePath,t):
        embeddings_index = self.loadEmbeddings(embeddingFilePath)
        # INITIALIZE EMBEDDINGS TO RANDOM VALUES
        embed_size = 300
        vocab_size = len(t.word_index) + 1
        sd = 1/np.sqrt(embed_size)  # Standard deviation to use
        embedding_matrix = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
        embedding_matrix = embedding_matrix.astype(np.float32)
        # create a weight matrix for words in training docs
        # embedding_matrix = zeros((vocab_size, 300))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                #### TODO bulunamayanlar unknownlar hepsi farklı random vector oluyor. Bu iyi bir şey değil tek bir UNK random vector olsun !!!
                print("################",str(word))
        return embedding_matrix