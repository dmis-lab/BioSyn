import numpy as np
import torch
import argparse
import logging
import time
import pdb
import os
import json
import random
from utils import (
    evaluate
)
from tqdm import tqdm
from src.biosyn import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    BioSyn
)

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    # Required
    # parser.add_argument('--model_dir', required=True,
    #                     help='Directory for pretrained model')
    parser.add_argument('--model_name_or_path', required=True,
                        help='name or path for dense encoder (e.g., bert-base-cased)')
    parser.add_argument('--train_dictionary_path', type=str, required=True,
                    help='train dictionary path')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--do_eval',  action="store_true")
    parser.add_argument('--eval_dictionary_path', type=str, required=False,
                    help='eval dictionary path')
    parser.add_argument('--eval_dir', type=str, required=False,
                    help='eval set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_sparse_weight',
                        default=0, type=float)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--draft', action="store_true")
    parser.add_argument('--do_cuiless', action="store_true")

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path, cuiless_token):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path,
            cuiless_token = cuiless_token
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x) 
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    cuiless_token = None
    if args.do_cuiless:
        cuiless_token = '[CUI-LESS]'

    # load dictionary and queries
    train_dictionary = load_dictionary(
        dictionary_path=args.train_dictionary_path,
        cuiless_token=cuiless_token
    )
    train_queries = load_queries(
        data_dir = args.train_dir, 
        filter_composite=True,
        filter_duplicate=True
    )

    if args.draft:
        train_dictionary = train_dictionary[:1000]
        train_queries = train_queries[:100]
    
    if args.do_eval:
        eval_dictionary = load_dictionary(
            dictionary_path=args.eval_dictionary_path,
            cuiless_token=cuiless_token
        )
        eval_queries = load_queries(
            data_dir=args.eval_dir,
            filter_composite=False,
            filter_duplicate=False
        )
        if args.draft:
            eval_dictionary = eval_dictionary[:1000]
            eval_queries = eval_queries[:100]

    # filter only names
    names_in_train_dictionary = train_dictionary[:,0] # [name, cui]
    names_in_train_queries = train_queries[:,0] # [name, cui, pmid]

    # names_in_train_dictionary = train_dictionary[:,0]
    # names_in_train_queries = train_queries[:,0]

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = BioSyn(
        max_length=args.max_length,
        use_cuda=args.use_cuda
    )

    encoder, tokenizer = biosyn.load_dense_encoder(
        model_name_or_path=args.model_name_or_path,
        cuiless_token= cuiless_token
    )

    # encoder, tokenizer = biosyn.load_bert(
    #     path=args.model_dir, 
    #     max_length=args.max_length,
    #     use_cuda=args.use_cuda,
    # )
    sparse_encoder = biosyn.train_sparse_encoder(corpus=names_in_train_dictionary)
    sparse_weight = biosyn.init_sparse_weight(
        initial_sparse_weight=args.initial_sparse_weight,
    )
    
    # load rerank model
    model = RerankNet(
        encoder = encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        sparse_weight=sparse_weight,
        use_cuda=args.use_cuda
    )
    
    # embed sparse representations for query and dictionary
    # Important! This is one time process because sparse represenation never changes.
    LOGGER.info("Sparse embedding")
    train_query_sparse_embeds = biosyn.embed_sparse(names=names_in_train_queries)
    train_dict_sparse_embeds = biosyn.embed_sparse(names=names_in_train_dictionary)
    train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=train_query_sparse_embeds, 
        dict_embeds=train_dict_sparse_embeds
    )
    train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=train_sparse_score_matrix, 
        topk=args.topk
    )

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = tokenizer,
        max_length = args.max_length,
        topk = args.topk, 
        d_ratio=args.dense_ratio,
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs,
        cuiless_token=cuiless_token
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    start = time.time()
    best_performance = 0.0
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        train_query_dense_embeds = biosyn.embed_dense(names=names_in_train_queries, show_progress=True)
        train_dict_dense_embeds = biosyn.embed_dense(names=names_in_train_dictionary, show_progress=True)
        train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_dense_embeds, 
            dict_embeds=train_dict_dense_embeds
        )
        train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=train_dense_score_matrix, 
            topk=args.topk
        )
        # replace dense candidates in the train_set
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        # train
        train_loss = train(args, data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)
        
        if args.do_eval:
            result_evalset = evaluate(
                biosyn=biosyn,
                eval_dictionary=eval_dictionary,
                eval_queries=eval_queries,
                topk=args.topk,
                score_mode="dense"
            )
            if best_performance < result_evalset['acc1']:
                best_performance = result_evalset['acc1']
                # save model best epoch
                biosyn.save_model(args.output_dir)
                output_file = os.path.join(args.output_dir,"predictions_eval.json")
                with open(output_file, 'w') as f:
                    json.dump(result_evalset, f, indent=2)
                print(f"best_performance={best_performance}")

        # # save model last epoch
        # if epoch == args.epoch:
        #     biosyn.save_model(args.output_dir)

    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
