import logging
import os
import collections
from .bert_utils import (
    BasicTokenizer, 
    WordpieceTokenizer
)

LOGGER = logging.getLogger(__name__)


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_pair(self, idx, word):
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def list_vocab(self):
        return list(self.word2idx.keys())


class BertTokenizer(object):

    def __init__(self,
                path,
                max_length=512,
                do_basic_tokenize=True,
                do_lower_case=True,
                never_split=("[UNK]", "[SEP]", "[PAD]", 
                             "[CLS]", "[MASK]")):
        super(BertTokenizer, self).__init__()
        LOGGER.info("BertTokenizer! max_length={}".format(max_length))
    
        # load bert vocab
        self.bert_vocab_file = os.path.join(path,"vocab.txt")
        if not os.path.isfile(self.bert_vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'.".format(self.bert_vocab_file))
        self.bert_vocab = dict(self.load_bert_vocab(self.bert_vocab_file))

        self.max_length = max_length
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.bert_vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.bert_vocab)

    def transform(self, words):
        split_tokens = []
        max_mention_length = self.max_length -2 # truncate, prepare [CLS] and [SEP]
        
        for word in words:
            sub_tokens = self.tokenize(word)
            if len(sub_tokens) > max_mention_length: # truncate, prepare [CLS] and [SEP]
                sub_tokens = sub_tokens[:max_mention_length]
            sub_tokens = ["[CLS]"] + sub_tokens + ["[SEP]"]
            
            #! add zero-padding
            sub_tokens += ['[PAD]'] * (self.max_length-len(sub_tokens))
            
            sub_tokens = [self.bert_vocab[t] for t in sub_tokens]
            split_tokens.append(sub_tokens)

        return split_tokens

    def tokenize(self,word):
        sub_tokens= []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(word):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    sub_tokens.append(sub_token)
        else:
            sub_tokens += self.wordpiece_tokenizer.tokenize(word)

        return sub_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.bert_vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.ids_to_tokens, ids)

    def load_bert_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def save_bert_vocab(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.bert_vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    LOGGER.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        LOGGER.info("Vocabulary saved in {}".format(vocab_file))
        return (vocab_file,)


def test():
    # Test bert tokenizer
    tokenizer = BertTokenizer()
    words = "Mr. Trump discussed Brexit with Mrs. May"
    print(words)
    tokens = tokenizer.tokenize(words)
    print(tokens)


if __name__ =="__main__":
    from tasks.tokenizer import Vocabulary
    test()
