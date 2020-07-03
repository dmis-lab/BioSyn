import argparse
import os
import pdb
import pandas as pd
import logging
from tqdm import tqdm
import numpy as np
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARN
)

def make_wordset(row):
    wordset = set()
    name = set([row['Name']])
    wordset = wordset.union(name)

    synonyms = row['Synonyms']
    if pd.notna(synonyms):
        synonyms = set(synonyms.split('|'))
        wordset = wordset.union(synonyms)
    return wordset

class StringConverter(dict):
    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return str

    def get(self, default=None):
        return str    

class CTDPreprocess(object):
    def __init__(self, _type, inpath, outpath):
        self.type = _type
        assert self.type in ['disease', 'chemical', 'gene',]
        assert self.type in inpath

        self.inpath = inpath
        self.outpath = outpath

        self.TSV_PROPERTIES ={
            'disease': {
                "usecols": [0, 1, 2, 7],
                "names":['Name', 'ID', 'AltIDs', 'Synonyms'],
            },
            'chemical': {
                "usecols": [0, 1, 7],
                "names":['Name', 'ID', 'Synonyms'],
            },
            'gene':  {
                "usecols": [1, 2, 3, 4],
                "names":['Name', 'ID', 'AltIDs', 'Synonyms'],
            },
        }
        
    def load_dictionary(self):
        tsv_property = self.TSV_PROPERTIES[self.type]

        dict_df = pd.read_csv(self.inpath, 
                            comment='#', 
                            header=None, 
                            usecols=tsv_property['usecols'], 
                            names=tsv_property['names'],
                            delimiter='\t',
                            converters=StringConverter())
        return dict_df

    def preprocess(self, outpath):
        dict_df = self.load_dictionary()
        outputs = []

        for row_idx, row in tqdm(dict_df.iterrows()):
            try:
                _id = row['ID']
                if 'altIDs' not in row or pd.isna(row['AltIDs']):
                    altIDs = ''
                else:
                    altIDs = row['AltIDs']
                if altIDs: ids = '|'.join([_id,altIDs])
                else: ids = _id
                if self.type == 'gene':
                    ids = "|".join(["NCBI:gene"+id for id in ids.split("|")])
                name = row['Name']
                synonyms = row['Synonyms'] if not pd.isna(row['Synonyms']) else ''
                # TODO! refactoring
                if synonyms: 
                    if name:
                        names = '|'.join([name,synonyms])
                    else:
                        names = synonyms
                else: 
                    if name: 
                        names = name
                    else:
                        continue
                output = '||'.join([ids,names])
                outputs.append(output)
            except Exception as ex:
                logging.warning(ex)
                pdb.set_trace()
            
        with open(self.outpath , 'w') as outfile:
            for output in outputs:
                outfile.write(output)
                outfile.write('\n')

def main(args):
    preprocessor = CTDPreprocess(_type=args.type, inpath=args.inpath, outpath=args.outpath)
    preprocessor.preprocess(args.outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True,
                    help='type of mention')
    parser.add_argument('--inpath', type=str, required=True,
                    help='input path')
    parser.add_argument('--outpath', type=str, required=True, 
                    help='output path')

    args = parser.parse_args()

    main(args)