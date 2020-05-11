import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)


class RerankNet(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, sparse_weight, use_cuda):

        LOGGER.info("RerankNet! learning_rate={} weight_decay={} sparse_weight={} use_cuda={}".format(
            learning_rate,weight_decay,sparse_weight,use_cuda
        ))
        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.sparse_weight = sparse_weight
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params' : self.sparse_weight, 'lr': 0.01, 'weight_decay': 0}], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        self.criterion = marginal_nll
        
    def forward(self, x):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        query_token, candidate_tokens, candidate_s_scores = x
        batch_size, topk, _ = candidate_tokens.shape
        if self.use_cuda:
            candidate_s_scores = candidate_s_scores.cuda()
        
        # dense embed for query and candidates
        query_embed = self.encoder(query_token).unsqueeze(dim=1)  # query : [batch_size, 1, hidden]
        candidate_tokens = self.reshape_candidates_for_encoder(candidate_tokens)
        candidate_embeds = self.encoder(candidate_tokens).view(batch_size, topk, -1)  # [batch_size, topk, hidden]
        
        # score dense candidates
        candidate_d_score = torch.bmm(query_embed, candidate_embeds.permute(0,2,1)).squeeze(1)
        score = self.sparse_weight * candidate_s_scores + candidate_d_score
        return score

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss = self.criterion(outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


def marginal_nll(score, target):
    """
    sum all scores among positive samples
    """
    predict = F.softmax(score, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)                   # sum all positive scores
    loss = loss[loss > 0]                     # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
    loss = -torch.log(loss)                   # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()                     # will return zero loss
    else:
        loss = loss.mean()
    return loss
