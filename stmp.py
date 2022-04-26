import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class stmp(nn.Module):
    def __init__(self):
        super(stmp, self).__init__()
        # 我设定embedding size 为 8
        self.embedding_size = 8
        # embedding 层
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # mlp层
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
    # 在推荐系统中 我们最需要的就是 序列长啥样 和 序列的长度
    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_emb(item_seq)
        # Ms Mt
        # 获得embed 最后一个输出
        last_inputs = self.gather_indexes(item_seq_emb, item_seq_len - 1)
        org_memory = item_seq_emb
        # 通用兴趣
        ms = torch.div(torch.sum(org_memory, dim=1), item_seq_len)
        # 最后的兴趣
        mt = last_inputs

        # mlp
        hs = self.tanh(self.mlp_a(ms))
        ht = self.tanh(self.mlp_b(mt))
        seq_output = hs * ht
        return seq_output