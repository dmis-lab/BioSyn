"""
Input: train/gold_xml

Output: (train/test)/*.concept, *.txt
"""

import os
import pdb
import glob
import argparse
import xml.etree.ElementTree as elemTree

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--input_dir', required=True,
                        help='input directory (train_xml or gold_xml)')
    parser.add_argument('--output_dir', type=str, required=True,
                    help='output directory')
    
    args = parser.parse_args()
    return args
    
# input_dirs = [
#     '/home/mujeen/works/biosyn/datasets/raw/tac2017adr/train_xml',
#     '/home/mujeen/works/biosyn/datasets/raw/tac2017adr/gold_xml',
# ]

# output_dirs = [
#     os.path.join('/home/mujeen/works/biosyn/datasets/tac2017adr', 'train'),
#     os.path.join('/home/mujeen/works/biosyn/datasets/tac2017adr', 'test'),
# ]

def parse_xml(file):
    doc = elemTree.parse(file)
    root = doc.getroot()

    # text
    text = ""
    sections = root.findall('./Text/Section')
    for section in sections:
        text += section.text

    # reaction nodes have all mention and id.
    mention2id = {}
    reactions = root.findall('./Reactions/Reaction')
    for reaction in reactions:
        mention = reaction.attrib['str'].lower()
        cuis = []
        for normalization in reaction.findall('Normalization'):
            if 'meddra_pt_id' in normalization.attrib:
                cuis.append(normalization.attrib['meddra_pt_id'])
        cuis = '|'.join(cuis)
        if mention not in mention2id:
            mention2id[mention] = cuis
        else:
            raise ValueError('mention({}) already have id({}) in mention2id dictionary.'.format(mention, id_))
    
    # mention node have section and span information.
    entity_mentions = [mention for mention in mention2id.keys()]
    entity_ids = [mention2id[mention] if mention2id[mention] else '-1' for mention in entity_mentions]
    id_name = list(zip(entity_ids, entity_mentions))

    return text, id_name

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.xml")))
    num_docs=0
    num_queries = 0
    for input_file in input_files:
        document_name = os.path.basename(input_file).split(".")[0]
        txtname = document_name + '.txt'
        conceptname = document_name + '.concept'

        text, id_names = parse_xml(input_file)

        # save text
        with open(os.path.join(output_dir,txtname) ,'w') as f:
            f.write(text.lower())
        
        # save entity
        with open(os.path.join(output_dir,conceptname) ,'w') as f:
            for cui, mention in id_names:
                f.write("-1||-1|-1||-1||{}||{}".format(mention, cui))
                f.write("\n")
                num_queries +=1
        num_docs+=1
        
    print("{} {} {}".format(output_dir, num_docs,num_queries))    

if __name__ == '__main__':
    args = parse_args()
    main(args)