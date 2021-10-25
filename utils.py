import csv
import json
import numpy as np
import pdb
from tqdm import tqdm

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(biosyn, eval_dictionary, eval_queries, topk):
    encoder = biosyn.get_dense_encoder()
    tokenizer = biosyn.get_dense_tokenizer()
    
    # embed dictionary
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:,0], show_progress=True)
    
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        
        dict_mentions = []
        for mention in mentions:
            mention_dense_embeds = biosyn.embed_dense(names=np.array([mention]))
            
            # get score matrix
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dict_dense_embeds
            )
            score_matrix = dense_score_matrix

            candidate_idxs = biosyn.retrieve_candidate(
                score_matrix = score_matrix, 
                topk = topk
            )
            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'cui':np_candidate[1],
                    'label':check_label(np_candidate[1],golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result

def evaluate(biosyn, eval_dictionary, eval_queries, topk):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions

    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(biosyn,eval_dictionary,eval_queries, topk)
    result = evaluate_topk_acc(result)
    
    return result