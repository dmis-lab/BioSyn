"""
Input: NCBI(train/development/test)set_corpus.txt
https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
Output: (train/dev/test)/*.concept, *.txt
"""

import os
import argparse

def main(args):
    input_file = args.input_file
    output_dir = args.output_dir

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
    for line in lines:
        line = line.strip()
        if '|t|' in line:
            title = line.split("|")[2]
        elif '|a|' in line:
            abstract = line.split("|")[2]
        elif '\t' in line:
            line = line.split("\t")
            if len(line) == 6:
                pmid, start, end, mention, _class, cui = line
            else:
                raise NotImplementedError()
            query = pmid + "||"+start +"|" + end + "||" + _class + "||" + mention + "||" + cui
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
                    default="./raw/ncbi-disease/NCBItrainset_corpus.txt",
                    help='path of input file')
    parser.add_argument('--output_dir', type=str,
                    default="./ncbi-disease/train", 
                    help='path of output directionary')

    args = parser.parse_args()
    
    main(args)