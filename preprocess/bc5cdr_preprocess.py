"""
Input: CDR_(Training/Development/Test)Set.PubTator.txt
https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
Output: (train/dev/test)/*.concept, *.txt
"""

import os
import argparse
from tqdm import tqdm

def main(args):
    input_file = args.input_file
    output_dir = args.output_dir
    _type = args.type

    # create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # read lines from raw file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    queries = []
    pmids = []
    lines = lines + ['\n']
    num_docs = 0
    num_queries = 0
    for line in tqdm(lines):
        line = line.strip()
        if '|t|' in line:
            title = line.split("|")[2]
        elif '|a|' in line:
            abstract = line.split("|")[2]
        elif '\t' in line:
            line = line.split("\t")
            if len(line) == 6:
                pmid, start, end, mention, _class, cui = line
            elif len(line) == 4: # CID
                continue
            elif len(line) == 7: # Composite mention
                pmid, start, end, mention, _class, cui, composite_mentions = line
                if composite_mentions.count("|") == cui.count("|"):
                    mention = composite_mentions
            query = pmid + "||"+start +"|" + end + "||" + _class + "||" + mention + "||" + cui
            if _class.lower()==_type.lower():
                queries.append(query)
        elif len(queries): 
            
            if pmid in pmids:
                print(pmid)
                queries = []
                title = ""
                abstract = ""
                continue
            context = title + "\n\n" + abstract + "\n"
            
            concept = "\n".join(queries) + "\n"
            output_context_file = os.path.join(output_dir, "{}.txt".format(pmid))
            output_concept_file = os.path.join(output_dir, "{}.concept".format(pmid))
            with open(output_context_file, 'w') as f:
                f.write(context)
            with open(output_concept_file, 'w') as f:
                f.write(concept)
                
            num_docs +=1
            num_queries += len(queries)
            pmids.append(pmid)
            queries = []
            title = ""
            abstract = ""
    
    print("{} {} {}".format(output_dir, num_docs,num_queries))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                    default="./raw/bc5cdr/CDR_TrainingSet.PubTator.txt",
                    help='path of input file')
    parser.add_argument('--output_dir', type=str,
                    default="./bc5cdr-disease/train", 
                    help='path of output directionary')
    parser.add_argument('--type', type=str, choices=["chemical", "disease"])

    args = parser.parse_args()
    
    main(args)