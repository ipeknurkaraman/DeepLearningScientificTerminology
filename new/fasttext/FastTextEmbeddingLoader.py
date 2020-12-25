import numpy as np


class FastTextEmbeddingLoader:
    def loadEmbeddings(self, embeddingFilePath):
        file = open(embeddingFilePath)
        vocab_and_vectors = {}
        # put words as dict indexes and vectors as words values
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            vocab_and_vectors[word] = vector
        return vocab_and_vectors

    def getEmbeddingMatrix(self, embeddingFilePath, t):
        vocab_and_vectors = self.loadEmbeddings(embeddingFilePath)
        word_index = t.word_index
        vocab_size=len(word_index) + 1
        embed_size=300
        sd = 1/np.sqrt(embed_size)  # Standard deviation to use
        embedding_matrix = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = vocab_and_vectors.get(word)
            # words that cannot be found will be set to 0
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
