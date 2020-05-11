
import os
import torch
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


class SparseEncoder(object):
    def __init__(self, use_cuda=False):
        self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        self.use_cuda = use_cuda

    def fit(self, train_corpus):
        self.encoder.fit(train_corpus)
        return self

    def transform(self, mentions):
        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec) # return torch float tensor
        if self.use_cuda:
            vec = vec.cuda()
        return vec

    def cuda(self):
        self.use_cuda = True

        return self
    
    def cpu(self):
        self.use_cuda = False
        return self

    def __call__(self, mentions):
        return self.transform(mentions)
    
    def vocab(self):
        return self.encoder.vocabulary_

    def save_encoder(self, path):
        with open(path, 'wb') as fout:
            pickle.dump(self.encoder, fout)
            logging.info("Sparse encoder saved in {}".format(path))

    def load_encoder(self, path):
        with open(path, 'rb') as fin:
            self.encoder =  pickle.load(fin)
            logging.info("Sparse encoder loaded from {}".format(path))

        return self