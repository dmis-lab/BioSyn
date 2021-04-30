"""
Input: CDR_(Training/Development/Test)Set.PubTator.txt
https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
Output: (train/dev/test)/*.concept, *.txt
"""

import os
import pdb

# input_files = [
#     './datasets/raw/bc5cdr/CDR_TrainingSet.PubTator.txt',
#     './datasets/raw/bc5cdr/CDR_DevelopmentSet.PubTator.txt',
#     './datasets/raw/bc5cdr/CDR_TestSet.PubTator.txt',
# ]

# disease_output_dirs = [
#     os.path.join('./datasets/bc5cdr-disease', 'train'),
#     os.path.join('./datasets/bc5cdr-disease', 'dev'),
#     os.path.join('./datasets/bc5cdr-disease', 'test'),
# ]

# chemical_output_dirs = [
#     os.path.join('./datasets/bc5cdr-chemical', 'train'),
#     os.path.join('./datasets/bc5cdr-chemical', 'dev'),
#     os.path.join('./datasets/bc5cdr-chemical', 'test'),
# ]

for input_file, disease_output_dir, chemical_output_dir in zip(input_files, disease_output_dirs, chemical_output_dirs):
    if not os.path.exists(disease_output_dir):
        os.makedirs(disease_output_dir)
    if not os.path.exists(chemical_output_dir):
        os.makedirs(chemical_output_dir)
        
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    
    disease_queries = []
    chemical_queries = []
    pmids = []
    lines = lines + ['\n']
    num_docs = 0
    num_disease_queries = 0
    num_chemical_queries = 0
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
            elif len(line) == 4: # CID
                continue
            elif len(line) == 7: # Composite mention
                pmid, start, end, mention, _class, cui, composite_mentions = line
                if composite_mentions.count("|") == cui.count("|"):
                    mention = composite_mentions
            query = pmid + "||"+start +"|" + end + "||" + _class + "||" + mention + "||" + cui
            if _class=="Chemical":
                chemical_queries.append(query)
            elif _class=="Disease":
                disease_queries.append(query)
        elif len(disease_queries) or len(chemical_queries): 
            if pmid in pmids:
                print(pmid)
                disease_queries = []
                chemical_queries = []
                title = ""
                abstract = ""
                continue
            context = title + "\n\n" + abstract + "\n"
            
            
            # disease
            disease_concept = "\n".join(disease_queries) + "\n"
            output_context_file = os.path.join(disease_output_dir, "{}.txt".format(pmid))
            output_concept_file = os.path.join(disease_output_dir, "{}.concept".format(pmid))
            with open(output_context_file, 'w') as f:
                f.write(context)
            with open(output_concept_file, 'w') as f:
                f.write(disease_concept)
                
            # chemical
            chemical_concept = "\n".join(chemical_queries) + "\n"
            output_context_file = os.path.join(chemical_output_dir, "{}.txt".format(pmid))
            output_concept_file = os.path.join(chemical_output_dir, "{}.concept".format(pmid))
            with open(output_context_file, 'w') as f:
                f.write(context)
            with open(output_concept_file, 'w') as f:
                f.write(chemical_concept)
                
            num_docs +=1
            num_chemical_queries += len(chemical_queries)
            num_disease_queries += len(disease_queries)
            pmids.append(pmid)
            disease_queries = []
            chemical_queries = []
            title = ""
            abstract = ""
            # pdb.set_trace()
    
    print("{} {} {}".format(disease_output_dir, num_docs,num_disease_queries))
    print("{} {} {}".format(chemical_output_dir, num_docs,num_chemical_queries))

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