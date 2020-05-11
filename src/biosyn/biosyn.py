import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from .tokenizer import BertTokenizer
from .sparse_encoder import SparseEncoder
from .bertmodel import BertModel
from .rerankNet import RerankNet

from transformers import (
    BertConfig
)

LOGGER = logging.getLogger()


class BioSyn(object):
    """
    Wrapper class for dense encoder and sparse encoder
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder = None
        self.sparse_encoder = None
        self.sparse_weight = None

    def init_sparse_weight(self, initial_sparse_weight, use_cuda):
        """
        Parameters
        ----------
        initial_sparse_weight : float
            initial sparse weight
        """
        if use_cuda:
            self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
        else:
            self.sparse_weight = nn.Parameter(torch.empty(1))
        self.sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight

        return self.sparse_weight

    def train_sparse_encoder(self, corpus):
        self.sparse_encoder = SparseEncoder().fit(corpus)

        return self.sparse_encoder

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def get_sparse_encoder(self):
        assert (self.sparse_encoder is not None)
        
        return self.sparse_encoder

    def get_sparse_weight(self):
        assert (self.sparse_weight is not None)
        
        return self.sparse_weight

    def save_model(self, path):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

        # save bert vocab
        self.tokenizer.save_bert_vocab(path)
        
        # save sparse encoder

        sparse_encoder_path = os.path.join(path,'sparse_encoder.pk')
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)
        
        sparse_weight_file = os.path.join(path,'sparse_weight.pt')
        torch.save(self.sparse_weight, sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(sparse_weight_file))

    def load_model(self, path, max_length=25, use_cuda=True):
        self.load_bert(path, max_length, use_cuda)
        self.load_sparse_encoder(path)
        self.load_sparse_weight(path)
        
        return self

    def load_bert(self, path, max_length, use_cuda):
        self.tokenizer = BertTokenizer(path=path, max_length=max_length)
        config = BertConfig.from_json_file(os.path.join(path, "config.json"))
        self.encoder = BertModel(path=path, config=config, use_cuda=use_cuda) # dense encoder

        return self.encoder, self.tokenizer

    def load_sparse_encoder(self, path):
        self.sparse_encoder = SparseEncoder().load_encoder(path=os.path.join(path,'sparse_encoder.pk'))

        return self.sparse_encoder

    def load_sparse_weight(self, path):
        sparse_weight_file = os.path.join(path, 'sparse_weight.pt')
        self.sparse_weight = torch.load(sparse_weight_file)

        return self.sparse_weight

    def get_score_matrix(self, query_embeds, dict_embeds, is_sparse=False):
        """
        Return score matrix

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def embed_sparse(self, names, show_progress=False):
        """
        Embedding data into sparse representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        sparse_embeds : np.array
            A list of sparse embeddings
        """
        batch_size=1024
        sparse_embeds = []
        
        if show_progress:
            iterations = tqdm(range(0, len(names), batch_size))
        else:
            iterations = range(0, len(names), batch_size)
        
        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def embed_dense(self, names, show_progress=False):
        """
        Embedding data into sparse representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=1024
        dense_embeds = []

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)
                
            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                batch_tokenized_names = self.tokenizer.transform(batch)
                batch_dense_embeds = self.encoder(batch_tokenized_names)
                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds