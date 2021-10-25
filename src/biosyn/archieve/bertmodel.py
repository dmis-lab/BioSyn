import os
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from transformers import BertModel as bm


class BertModel(nn.Module):

    def __init__(self, path, config, use_cuda):
        logging.info("BertModel! use_cuda={}".format(use_cuda))
        super(BertModel, self).__init__()
        self.load_pretrained(path, config) # load pretrained
        self.embed_dim = 768
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()

    def load_pretrained(self, path, config):
        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('bert.'):
                k = k.replace('bert.', '')
                new_state_dict[k]=v
            elif k.startswith('cls.'):
                continue
            else:
                new_state_dict[k]=v
        
        self.model = bm(config)
        self.model.load_state_dict(new_state_dict)

    def save_pretrained(self, path):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(path)

    def forward(self, subwords):
        """
        @ input: variable of tensor with shape [batch, len(subwords)]
        @ output: variable of tensor with shape [batch, word_embed_dim]
        """
        input_ids = torch.LongTensor(subwords)
        input_masks= input_ids.clone()
        input_masks[input_masks>0] = 1 # masking
        if self.use_cuda:
            input_ids = input_ids.cuda()
            input_masks = input_masks.cuda()
        last_hidden_state, _ = self.model(input_ids=input_ids,attention_mask=input_masks)
        x = last_hidden_state[:,0] # CLS token representation
        return x

    def cuda(self):
        self.use_cuda = True
        self.model = self.model.cuda()
        return self

    def cpu(self):
        self.use_cuda = False
        self.model = self.model.cpu()
        return self