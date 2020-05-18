import argparse
import os
from tqdm import tqdm
import pdb

class MedDRAPreprocess():
    """
    Make training dictionary pair
    """
    def __init__(self, hlgt_path, hlt_path, pt_path, llt_path):
        self.hlgt_path = hlgt_path
        self.hlt_path = hlt_path
        self.pt_path =pt_path
        self.llt_path = llt_path

    def load_dictionary(self):
        """ 
        ! hlgt, hlt, pt => need to extract id and name
        format id$name$$$$$$$$

        ! llt => need to extract pt_id and name
        format id$name$pt_id$$$$$$$
        """

        dictionary = {}
        # hlgt
        with open(self.hlgt_path, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split("$")
                _id = line[0]
                _name = line[1]
                if _id not in dictionary.keys():
                    dictionary[_id] = _name  
                else:
                    dictionary[_id] = dictionary[_id] + "|" + _name
        # hlt
        with open(self.hlt_path, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split("$")
                _id = line[0]
                _name = line[1]
                if _id not in dictionary.keys():
                    dictionary[_id] = _name  
                else:
                    dictionary[_id] = dictionary[_id] + "|" + _name

        # pt
        with open(self.pt_path, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split("$")
                _id = line[0]
                _name = line[1]
                if _id not in dictionary.keys():
                    dictionary[_id] = _name  
                else:
                    dictionary[_id] = dictionary[_id] + "|" + _name

        # llt
        with open(self.llt_path, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split("$")
                _id = line[2]
                _name = line[1]
                if _id not in dictionary.keys():
                    dictionary[_id] = _name  
                else:
                    names = dictionary[_id].split("|")
                    names.append(_name)
                    names = "|".join(list(set(names)))
                    dictionary[_id] = names

        list_dictionary = [[k,v] for k,v in dictionary.items()]
        return list_dictionary

    def make_ID_mention_map(self, out):
        dictionary = self.load_dictionary()
        with open(out, 'w') as outfile:
            for row in dictionary:
                outfile.write("||".join(row))
                outfile.write('\n')
                
def parse_args(debug=False):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Data Preprocess')

    # Required
    parser.add_argument('--hlgt_path', required=True)
    parser.add_argument('--hlt_path', required=True)
    parser.add_argument('--pt_path', required=True)
    parser.add_argument('--llt_path', required=True)
    parser.add_argument('--output_path', required=True)

    args = parser.parse_args()
    
    return args

def main(args):
    meddra_preprocesser = MedDRAPreprocess(
        hlgt_path=args.hlgt_path,
        hlt_path=args.hlt_path,
        pt_path=args.pt_path,
        llt_path=args.llt_path
    )
    
    meddra_preprocesser.make_ID_mention_map(out=args.output_path)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
