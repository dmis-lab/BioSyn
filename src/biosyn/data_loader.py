import re
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)


class QueryDataset(Dataset):

    def __init__(self, data_dir, 
                filter_composite=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, filter_composite, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                concept = concept.split("||")
                mention = concept[3].strip()
                cui = concept[4].strip()
                is_composite = (cui.replace("+","|").count("|") > 0)

                if filter_composite and is_composite:
                    continue
                else:
                    data.append((mention,cui))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)
        
    def load_data(self, dictionary_path):
        name_cui_map = {}
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                cui, name = line.split("||")
                data.append((name,cui))
        
        data = np.array(data)
        return data


class CandidateDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, queries, dicts, tokenizer, topk, d_ratio, s_score_matrix, s_candidate_idxs):
        """
        Retrieve top-k candidates based on sparse/dense embedding

        Parameters
        ----------
        queries : list
            A list of tuples (name, id)
        dicts : list
            A list of tuples (name, id)
        tokenizer : BertTokenizer
            A BERT tokenizer for dense embedding
        topk : int
            The number of candidates
        d_ratio : float
            The ratio of dense candidates from top-k
        s_score_matrix : np.array
        s_candidate_idxs : np.array
        """
        LOGGER.info("CandidateDataset! len(queries)={} len(dicts)={} topk={} d_ratio={}".format(
            len(queries),len(dicts), topk, d_ratio))
        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.topk = topk
        self.n_dense = int(topk * d_ratio)
        self.n_sparse = topk - self.n_dense
        self.tokenizer = tokenizer

        self.s_score_matrix = s_score_matrix
        self.s_candidate_idxs = s_candidate_idxs
        self.d_candidate_idxs = None

    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs
    
    def __getitem__(self, query_idx):
        assert (self.s_candidate_idxs is not None)
        assert (self.s_score_matrix is not None)
        assert (self.d_candidate_idxs is not None)

        query_name = self.query_names[query_idx]
        query_token = self.tokenizer.transform([query_name])

        # combine sparse and dense candidates as many as top-k
        s_candidate_idx = self.s_candidate_idxs[query_idx]
        d_candidate_idx = self.d_candidate_idxs[query_idx]
        
        # fill with sparse candidates first
        topk_candidate_idx = s_candidate_idx[:self.n_sparse]
        
        # fill remaining candidates with dense
        for d_idx in d_candidate_idx:
            if len(topk_candidate_idx) >= self.topk:
                break
            if d_idx not in topk_candidate_idx:
                topk_candidate_idx = np.append(topk_candidate_idx,d_idx)
        
        # sanity check
        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))
        
        candidate_names = [self.dict_names[candidate_idx] for candidate_idx in topk_candidate_idx]
        candidate_s_scores = self.s_score_matrix[query_idx][topk_candidate_idx]
        labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)
        query_token = np.array(query_token).squeeze()

        candidate_tokens = self.tokenizer.transform(candidate_names)
        candidate_tokens = np.array(candidate_tokens)
        
        return (query_token, candidate_tokens, candidate_s_scores), labels

    def __len__(self):
        return len(self.query_names)

    def check_label(self, query_id, candidate_id_set):
        label = 0
        query_ids = query_id.split("|")
        """
        All query ids should be included in dictionary id
        """
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        return label

    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels