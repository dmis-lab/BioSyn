"""
input dictionary: cui||synonyms
output dictionary: cui||name
"""

import pdb
import argparse
import glob
import os
from tqdm import tqdm
from biosyn import (
    TextPreprocess
)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    
    # Required
    parser.add_argument('--input_dictionary_path', type=str, required=True,
                        help='path of raw dictionary')
    parser.add_argument('--output_dictionary_path', type=str, required=True,
                        help='path of processed dictionary')

    # Append more terms from dataset
    parser.add_argument('--additional_data_dir', type=str, default=None,
                        help='path of additional dataset to append')
    
    # Text Preprocessing
    parser.add_argument('--lowercase',  action="store_true")
    parser.add_argument('--remove_punctuation',  action="store_true")

    args = parser.parse_args()

    return args

def convert_dictionary_to_name2cui(dictionary, text_preprocessor=None):
    """
    convert dictionary to name2cui to get unique names and the corresponding cuis
    
    Note! For names which are duplicate, concat the cui with '|' delimiter. 
    
    Parameters
    ----------
    dictionary : list
        A list of terms (cui||names)
    
    Returns
    -------
    name2cui : dict
        key is a name, value is a cui
    """

    name2cui = {}
    for term in tqdm(dictionary, total=len(dictionary)):
        cui, names = term.split("||")
        names = names.strip().split("|")
        for name in names:
            if text_preprocessor:
                name = text_preprocessor.run(name)
                
            # If new name, make a new name-cui pair
            if name not in name2cui:
                name2cui[name] = cui
            # If existing name and new cui with it, concat the cui
            elif name in name2cui and cui not in name2cui[name]:
                name2cui[name] += "|"+cui
            # If existing name and existing cui with it, skip
            else:
                pass
    
    return name2cui

def main(args):
    # load dictionary
    with open(args.input_dictionary_path, 'r') as f:
        dictionary = f.readlines()
        
    # load text preprocessor
    text_preprocessor = TextPreprocess(
        lowercase=args.lowercase, 
        remove_punctuation=args.remove_punctuation, 
    )
    
    # convert dictionary to name2cui to get unique names and the corresponding cuis
    name2cui = convert_dictionary_to_name2cui(
        dictionary=dictionary,
        text_preprocessor=text_preprocessor
    )
    
    # append terms in dataset to name2cui
    if args.additional_data_dir:
        concept_files = glob.glob(os.path.join(args.additional_data_dir, "*.concept"))
        for concept_file in concept_files:
            with open(concept_file, 'r') as f:
                train_examples = f.readlines()
            
            for train_examples in train_examples:
                _, _, _, name, cui = train_examples.strip().split("||")
                name = text_preprocessor.run(name)
                
                # If new name, make a new name-cui pair
                if name not in name2cui:
                    name2cui[name] = cui
                    # print("new name='{}' cui='{}'".format(name,cui))
                # If existing name and new cui with it, concat the cui
                elif name in name2cui and cui not in name2cui[name]:
                    name2cui[name] += "|"+cui
                    # print("existing name='{}' new cui='{}'".format(name,cui))
                # If existing name and existing cui with it, skip
                else:
                    pass
                
    print("total number of unique names={}".format(len(name2cui)))
    
    # save processed_dictionary
    # cui||name
    if not os.path.exists(os.path.dirname(args.output_dictionary_path)):
        os.makedirs(os.path.dirname(args.output_dictionary_path))
    with open(args.output_dictionary_path, 'w') as f:
        for name, cui in name2cui.items():
            f.write("{}||{}\n".format(cui, name))
    
                
if __name__ =='__main__':
    args = parse_args()
    main(args)