import torch
from attention_model import *

def generate_instance(size, prize_type):
    # Details see paper
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.,
        # 34: 39.
        34: 3.8
    }

    loc = torch.FloatTensor(size, 5).uniform_(0, 1)
    depot = torch.FloatTensor(5).uniform_(0, 1)
    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size, ))) / 100.
    elif prize_type == 'falcon':
        t = torch.ones(size) * 0.3333
        prize = torch.bernoulli(t)
        prize[prize==0] = 0.1
        prize[prize==1] = 0.3
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    return {
        'loc': loc,
        # Uniform 1 - 9, scaled by capacities
        'prize': prize,
        'depot': depot,
        'max_length': torch.tensor(MAX_LENGTHS[size])
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=3,
    mask_inner=True,
    mask_logits=True,
    normalization='batch',
    tanh_clipping=10.,
    checkpoint_encoder=False,
    shrink_size=None
).to(device)