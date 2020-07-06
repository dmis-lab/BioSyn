import argparse
import os
import pdb
import pickle
import tornado.web
import tornado.ioloop
import tornado.autoreload
import logging
import json

from src.biosyn import (
    DictionaryDataset,
    BioSyn,
    TextPreprocess
)

logging.basicConfig(
    filename='.server.log',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='BioSyn Demo')

# Required
parser.add_argument('--model_dir', required=True, help='Directory for model')

# Settings
parser.add_argument('--port', type=int, default=8888, help='port number')
parser.add_argument('--show_predictions',  action="store_true")
parser.add_argument('--dictionary_path', type=str, default=None, help='dictionary path')
parser.add_argument('--use_cuda',  action="store_true")

args = parser.parse_args()

def cache_or_load_dictionary():
    dictionary_name = os.path.splitext(os.path.basename(args.dictionary_path))[0]
    
    cached_dictionary_path = os.path.join(
        './tmp',
        "cached_{}.pk".format(dictionary_name)
    )

    # If exist, load the cached dictionary
    if os.path.exists(cached_dictionary_path):
        with open(cached_dictionary_path, 'rb') as fin:
            cached_dictionary = pickle.load(fin)
        print("Loaded dictionary from cached file {}".format(cached_dictionary_path))

        dictionary, dict_sparse_embeds, dict_dense_embeds = (
            cached_dictionary['dictionary'],
            cached_dictionary['dict_sparse_embeds'],
            cached_dictionary['dict_dense_embeds'],
        )

    else:
        dictionary = DictionaryDataset(dictionary_path = args.dictionary_path).data
        dictionary_names = dictionary[:,0]
        dict_sparse_embeds = biosyn.embed_sparse(names=dictionary_names, show_progress=True)
        dict_dense_embeds = biosyn.embed_dense(names=dictionary_names, show_progress=True)
        cached_dictionary = {
            'dictionary': dictionary,
            'dict_sparse_embeds' : dict_sparse_embeds,
            'dict_dense_embeds' : dict_dense_embeds
        }

        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')
        with open(cached_dictionary_path, 'wb') as fin:
            pickle.dump(cached_dictionary, fin)
        print("Saving dictionary into cached file {}".format(cached_dictionary_path))

    return dictionary, dict_sparse_embeds, dict_dense_embeds

def normalize(mention):
    # preprocess mention
    mention = TextPreprocess().run(mention)

    # embed mention
    mention_sparse_embeds = biosyn.embed_sparse(names=[mention])
    mention_dense_embeds = biosyn.embed_dense(names=[mention])

    # calcuate score matrix and get top 1
    sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=mention_sparse_embeds,
        dict_embeds=dict_sparse_embeds
    )
    dense_score_matrix = biosyn.get_score_matrix(
        query_embeds=mention_dense_embeds,
        dict_embeds=dict_dense_embeds
    )
    sparse_weight = biosyn.get_sparse_weight().item()
    hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
    hybrid_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix = hybrid_score_matrix, 
        topk = 10
    )
    
    # get predictions from dictionary
    predictions = dictionary[hybrid_candidate_idxs].squeeze(0)
    output = {
        'predictions' : []
    }

    for prediction in predictions:
        predicted_name = prediction[0]
        predicted_id = prediction[1]
        output['predictions'].append({
            'name': predicted_name,
            'id': predicted_id
        })

    return output

# load biosyn model
biosyn = BioSyn().load_model(
    path=args.model_dir,
    max_length=25,
    use_cuda=args.use_cuda
)

# cache or load dictionary
dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary()
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("./template/index.html")

class NormalizeHandler(tornado.web.RequestHandler):
    def get(self):
        string = self.get_argument('string', '')
        logging.info('get!{}'.format({
            'string':string,
        }))
        self.set_header("Content-Type", "application/json")    
        output = normalize(mention=string)
        
        self.write(json.dumps(output))
    
def make_app():
    settings={
        'debug':True
    }
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/normalize/", NormalizeHandler),
        (r'/semantic/(.*)', tornado.web.StaticFileHandler, {'path': './semantic'}),
        (r'/images/(.*)', tornado.web.StaticFileHandler, {'path': './images'}),
    ],**settings)


if __name__ == '__main__':
    logging.info('Starting biosyn server at http://localhost:{}'.format(args.port))       
    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
