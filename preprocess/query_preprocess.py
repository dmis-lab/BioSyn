import argparse
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import pdb

from biosyn import (
    Abbr_resolver,
    TextPreprocess
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_concept(path):
    result = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            pmid, span, mention_type, mention, cui = line.split('||')
            result.append([pmid, span, mention_type, mention, cui])
    return result

def write_concept(path, concept):
    with open(path, 'w') as f:
        for line in concept:
            line = '||'.join(line) + '\n'
            f.write(line)

def apply_basic_preprocess(concept, text_preprocessor):
    result = []
    for pmid, span, mention_type, mention, cui in concept:
        result_mention = text_preprocessor.run(mention)
        result.append([pmid, span, mention_type, result_mention, cui])
    return result

def apply_abbr_dict(concept, abbr_dict):
    result = []
    for pmid, span, mention_type, mention, cui in concept:
        prev_mention = mention
        while True:
            mention_tokens = prev_mention.split()
            result_tokens = []
            for token in mention_tokens:
                token = token.strip()

                if '/' in token:
                    _slash_result = []
                    for t in token.split('/'):
                        t = t.strip()
                        t = abbr_dict.get(t, t)
                        _slash_result.append(t)
                    token = '/'.join(_slash_result)

                if token.endswith(','):
                    token = token.replace(',', '')
                    abbr_dict : dict
                    token = abbr_dict.get(token, token)
                    token += ','
                else:
                    token = abbr_dict.get(token, token)
                result_tokens.append(token)
            result_mention = ' '.join(result_tokens)
            if result_mention == prev_mention:
                break
            else:
                prev_mention = result_mention
        result.append([pmid, span, mention_type, result_mention, cui])
    return result

def split_composite_cui(cui):
    if '|' in cui:
        composite_symbol = '|'
        cui_list = cui.split('|')
    elif '+' in cui:
        composite_symbol = '+'
        cui_list = cui.split('+')
    else:
        composite_symbol = ''
        cui_list = [cui]
    return cui_list, composite_symbol

def load_cui_set(path):
    cui_set = set()
    with open(path, 'r') as f:
        for line in f:
            cui, *_ = line.strip().split('||')
            for c in cui.replace('+', '|').split('|'):
                c = c.strip()
                cui_set.add(c)
    return cui_set


def filter_cuiless(concept, cui_set):
    result = []
    for pmid, span, mention_type, mention, cui in concept:
        cui = cui.replace('OMIM:', '').replace('MESH:', '')
        cui_list, _ = split_composite_cui(cui)
        cui_list = [cui.strip() for cui in cui_list]
        concept_cui_set = set(cui_list)
        cui_less_set = concept_cui_set - cui_set
        if len(cui_less_set) == 0:
            result.append([pmid, span, mention_type, mention, cui])
    return result

def composite_resolve(mention, cui):
    mention = mention.strip()

    # filter auxiliary cui
    cui_list, composite_symbol = split_composite_cui(cui)
    all_omim = all(['OMIM' in cui for cui in cui_list])
    all_mesh = all(['OMIM' not in cui for cui in cui_list])
    is_cui_mixed = not (all_omim or all_mesh)
    if is_cui_mixed:
        cui_list = filter(lambda cui: 'OMIM' not in cui, cui_list)
    cui_list = list(cui_list)

    mention_tokens = mention.split()
    if len(cui_list) == 1:
        return [mention], ''.join(cui_list)

    prefix_of_pattern = re.compile(
        "(?P<prefix>[a-zA-Z-]+) of (both )?(the )?(?P<suffix1>([a-zA-Z-]+ )+)(and|or) (the )?(?P<suffix2>([a-zA-Z-]+ ?)+)"
    )
    nested_and_pattern = re.compile(
        "(?P<prefix_list>([a-zA-Z-]+,? )+)(and|or|and/or) (?P<prefix_last>[a-zA-Z-]+ and [a-zA-Z-]+) (?P<stem>.*)"
    )
    trivial_pattern = re.compile(
        "(?P<prefix_list>([a-zA-Z-]+,? )+)(and|or|and/or) (?P<prefix_last>(the )?[a-zA-Z-]+) (?P<stem>.*)"
    )
    slash_pattern = re.compile(
        "(?P<prefix>(.* )*)(?P<composite1>.*)\/(?P<composite2>.*)?(?P<suffix>( .*)*)"
    )
    prefix_of_match = prefix_of_pattern.match(mention)
    nested_and_match = nested_and_pattern.match(mention)
    trivial_match = trivial_pattern.match(mention)
    slash_match = slash_pattern.fullmatch(mention)

    if prefix_of_match:
        match = prefix_of_match
        prefix = match.group('prefix')
        mention1 = ' '.join([prefix, 'of', match.group('suffix1').strip()])
        mention2 = ' '.join([prefix, 'of', match.group('suffix2').strip()])
        return [mention1, mention2], composite_symbol.join(cui_list)
    elif nested_and_match:
        match = nested_and_match
        stem = match.group('stem')
        prefix_list = match.group('prefix_list').strip().split(',')
        prefix_list.append(match.group('prefix_last'))
        prefix_list = filter(len, prefix_list)  # filter zero len prefix
        mention_list = [prefix.strip() + ' ' + stem for prefix in prefix_list]
        return mention_list, composite_symbol.join(cui_list)
    elif trivial_match:
        match = trivial_match
        stem = match.group('stem')
        prefix_list = match.group('prefix_list').strip().split(',')
        prefix_list.append(match.group('prefix_last'))
        prefix_list = filter(len, prefix_list)  # filter zero len prefix
        mention_list = [prefix.strip() + ' ' + stem for prefix in prefix_list]
        return mention_list, composite_symbol.join(cui_list)
    elif slash_match and 'and/or' not in mention:
        match = slash_match
        prefix = slash_match.group('prefix').strip()
        suffix = slash_match.group('suffix').strip()
        composite_list = [slash_match.group('composite1').strip(), slash_match.group('composite2').strip()]
        mention_list = [prefix + ' ' + composite + ' ' + suffix for composite in composite_list]
        mention_list = [mention.strip() for mention in mention_list]
        return mention_list, composite_symbol.join(cui_list)
    elif len(cui_list) == len(mention_tokens):
        return mention_tokens, composite_symbol.join(cui_list)
    else:  # if composite mention doesn't have valid number of cui, don't resolution.
        return [mention], composite_symbol.join(cui_list)


def apply_composite_resolve(concept):
    result = []
    for pmid, span, mention_type, mention, cui in concept:
        if "|" in mention: # already resolved composite mention
            result.append([pmid, span, mention_type, mention, cui])
            continue
        mention_list, cui = composite_resolve(mention, cui)
        cui_list, composite_symbol = split_composite_cui(cui)
        if len(cui_list) > 1 and len(mention_list) == len(cui_list):
            result.append([pmid, span, mention_type, '|'.join(mention_list), composite_symbol.join(cui_list)])
        else:
            result.append([pmid, span, mention_type, mention, cui])
    return result

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Data Preprocess')

    # Required
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    # Text Preprocessing
    parser.add_argument('--ab3p_path',  type=str, default=None, help='abbreviation resolution module')
    parser.add_argument('--typo_path',  type=str, default=None)
    parser.add_argument('--dictionary_path',  type=str, default=None)
    parser.add_argument('--remove_cuiless',  action="store_true")
    parser.add_argument('--resolve_composites',  action="store_true")
    parser.add_argument('--lowercase',  type=str2bool, default=True)
    parser.add_argument('--remove_punctuation',  type=str2bool, default=True)

    args = parser.parse_args()
    return args


def main(args):
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    dataset_name = input_dir.parent.stem
    output_dir.mkdir(exist_ok=True)
    input_files = input_dir.iterdir()
    input_files = filter(lambda path: path.suffix == '.concept', input_files)
    input_files = map(lambda path: path.stem, input_files)
    input_files = sorted(input_files)

    cui_set = None

    abbr_resolver = Abbr_resolver(
        ab3p_path=args.ab3p_path
    )
    text_preprocessor = TextPreprocess(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        ignore_punctuations='|',
        typo_path=args.typo_path
    )
    
    num_queries = 0
    for input_file in tqdm(input_files):
        concept_file = input_dir/(input_file+'.concept')
        txt_file = input_dir/(input_file+'.txt')
        output_path = output_dir/(input_file+'.concept')

        concept = parse_concept(concept_file)

        # apply abbreviation resolve
        abbr_dict = abbr_resolver.resolve(txt_file)
        concept = apply_abbr_dict(concept, abbr_dict)

        # apply composition resolve
        if args.resolve_composites:
            concept = apply_composite_resolve(concept)
            
        # apply basic preprocess
        concept = apply_basic_preprocess(concept, text_preprocessor)
        
        # remove cuiless
        if args.remove_cuiless:
            if cui_set is None:
                dict_path = args.dictionary_path
                cui_set = load_cui_set(dict_path)
            concept = filter_cuiless(concept, cui_set)
        num_queries += len(concept)
        write_concept(output_path, concept)
    
    print("total number of queries={}".format(num_queries))

if __name__ == '__main__':
    args = parse_args()
    main(args)
