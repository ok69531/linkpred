#%%
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, GAE


#%%
seed = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
''' train/validation/test data split '''
dataset = torch.load('dataset/homo_ctd_graph.pt')

### masking 생성
num_edges = dataset.edge_index.shape[1]
all_idx = list(range(num_edges))
random.shuffle(all_idx)

frac_train = 0.8
frac_valid = 0.1
train_idx = all_idx[: int(frac_train * num_edges)]
valid_idx = all_idx[int(frac_train * num_edges) : int(frac_train * num_edges) + int(frac_valid * num_edges)]
test_idx = all_idx[int(frac_train * num_edges) + int(frac_valid * num_edges): ]

len(train_idx) + len(valid_idx) + len(test_idx) == num_edges

# # valid, test data는 chem_cause_dis 엣지에서 train 데이터를 제외하고 각각 40/60%
# edge_type_num = torch.unique(dataset.edge_type, return_counts = True)
# chem_cau_dis_num = edge_type_num[1][0].item()
# chem_cau_dis_idx = list(range(chem_cau_dis_num))
# random.shuffle(chem_cau_dis_idx)

# is_in_train = [train_idx[i] for i in range(len(train_idx)) if train_idx[i] < chem_cau_dis_num ]
# chem_cau_dis_idx = [i for i in chem_cau_dis_idx if i not in is_in_train]

# frac_valid = 0.4
# valid_idx = chem_cau_dis_idx[: int(frac_valid * len(chem_cau_dis_idx))]
# test_idx = chem_cau_dis_idx[int(frac_valid * len(chem_cau_dis_idx)):]


train_mask = torch.zeros(num_edges, dtype = torch.bool)
train_mask[train_idx] = True

val_mask = torch.zeros(num_edges, dtype = torch.bool)
val_mask[valid_idx] = True

test_mask = torch.zeros(num_edges, dtype = torch.bool)
test_mask[test_idx] = True


### 데이터 분할
num_relations = dataset.num_relations
edge_index = dataset.edge_index[:, train_mask]
edge_type = dataset.edge_type[train_mask]

edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
edge_type = torch.cat([edge_type, edge_type + num_relations])
num_relations = num_relations * 2

train_data = Data(x = torch.tensor(dataset.x, dtype = torch.int), 
                  edge_index=edge_index, edge_type=edge_type, 
                  num_nodes=dataset.num_nodes, num_relations = num_relations,
                  target_edge_index=dataset.edge_index[:, train_mask],
                  target_edge_type=dataset.edge_type[train_mask])
valid_data = Data(x = torch.tensor(dataset.x, dtype = torch.int), 
                  edge_index=edge_index, edge_type=edge_type, 
                  num_nodes=dataset.num_nodes, num_relations = num_relations,
                  target_edge_index=dataset.edge_index[:, val_mask],
                  target_edge_type=dataset.edge_type[val_mask])
test_data = Data(x = torch.tensor(dataset.x, dtype = torch.int), 
                 edge_index=edge_index, edge_type=edge_type, 
                 num_nodes=dataset.num_nodes, num_relations = num_relations,
                 target_edge_index=dataset.edge_index[:, test_mask],
                 target_edge_type=dataset.edge_type[test_mask])


#%%
class RGCNet(nn.Module):
    def __init__(self, num_node, num_relation, emb_dim, num_layers = 3, num_bases = 30, dropout = 0):
        super(RGCNet, self).__init__()
        
        self.dropout = dropout
        self.node_embed = nn.Embedding(num_node, emb_dim)    
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(emb_dim, emb_dim, num_relation, num_bases))
        
    def forward(self, x, edge_index, edge_type):
        x = self.node_embed(x)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p = self.dropout, training = self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        
        return x


class MLPLinkPredictor(nn.Module):
    def __init__(self, emb_dim, out_dim, num_layers = 3, dropout = 0):
        super(MLPLinkPredictor, self).__init__()
        
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(emb_dim, emb_dim))
        self.layers.append(nn.Linear(emb_dim, out_dim))
        
    def forward(self, head, tail):
        x = head * tail
        for lin in self.layers[:-1]:
            x = lin(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p = self.dropout, training = self.training)
        score = self.layers[-1](x)
        
        return score


#%%
def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ), device=neg_edge_index.device)
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ), device=neg_edge_index.device)
        
    return neg_edge_index
# negative_sampling(test_data.target_edge_index, dataset.num_nodes)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.target_edge_index, data.target_edge_type)

    src, dst = data.target_edge_index
    pos_out = model.decode(z[src], z[dst])

    neg_src, neg_dst = negative_sampling(data.target_edge_index, data.num_nodes)
    neg_out = model.decode(z[neg_src], z[neg_dst])

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    loss = nn.functional.binary_cross_entropy_with_logits(out, gt)
    # reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    # loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def evaluation(model, data):
    model.eval()
    
    z = model.encode(data.x, data.edge_index, data.edge_type)

    mrr, hits1, hits3, hits10 = compute_metric(z, data.target_edge_index, data.target_edge_type)

    return mrr, hits1, hits3, hits10


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_metric(z, edge_index, edge_type):
    ranks = []
    for i in range(edge_type.numel()):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail = torch.tensor(random.sample(range(dataset.num_nodes), 500))
        dst_is_in_neg = torch.where(tail == dst)[0]
        if len(dst_is_in_neg) == 0:
            tail = torch.cat([torch.tensor([dst]), tail])
        else:
            tail = torch.cat([tail[:dst_is_in_neg], tail[dst_is_in_neg+1:]])
            add_tail = torch.randint(0, dataset.num_nodes, (1,))
            tail = torch.cat([torch.tensor([dst]), tail, add_tail])
    
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        # eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z[eval_edge_index[0]], z[eval_edge_index[1]])
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head = torch.tensor(random.sample(range(dataset.num_nodes), 500))
        src_is_in_neg = torch.where(head == src)[0]
        if len(src_is_in_neg) == 0:
            head = torch.cat([torch.tensor([src]), head])
        else:
            head = torch.cat([head[:src_is_in_neg], head[src_is_in_neg+1:]])
            add_head = torch.randint(0, dataset.num_nodes, (1,))
            head = torch.cat([torch.tensor([src]), head, add_head])
        
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        # eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z[eval_edge_index[0]], z[eval_edge_index[1]])
        rank = compute_rank(out)
        ranks.append(rank)

    mrr = (1. / torch.tensor(ranks, dtype=torch.float)).mean()
    hits1 = (torch.tensor(ranks) <= 1).to(torch.float32).mean()
    hits3 = (torch.tensor(ranks) <= 3).to(torch.float32).mean()
    hits10 = (torch.tensor(ranks) <= 10).to(torch.float32).mean()
    
    return mrr, hits1, hits3, hits10



# @torch.no_grad()
# def compute_metric(z, edge_index, edge_type):
#     ranks = []
#     for i in range(edge_type.numel()):
#         (src, dst), rel = edge_index[:, i], edge_type[i]

#         # Try all nodes as tails, but delete true triplets:
#         tail_mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
#         for (heads, tails), types in [
#             (train_data.target_edge_index, train_data.target_edge_type),
#             (valid_data.target_edge_index, valid_data.target_edge_type),
#             (test_data.target_edge_index, test_data.target_edge_type),
#         ]:
#             tail_mask[tails[(heads == src) & (types == rel)]] = False

#         tail = torch.arange(dataset.num_nodes)[tail_mask]
#         tail = torch.cat([torch.tensor([dst]), tail])
#         head = torch.full_like(tail, fill_value=src)
#         eval_edge_index = torch.stack([head, tail], dim=0)
#         eval_edge_type = torch.full_like(tail, fill_value=rel)

#         out = model.decode(z[eval_edge_index[0]], z[eval_edge_index[1]])
#         rank = compute_rank(out)
#         ranks.append(rank)

#         # Try all nodes as heads, but delete true triplets:
#         head_mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
#         for (heads, tails), types in [
#             (train_data.target_edge_index, train_data.target_edge_type),
#             (valid_data.target_edge_index, valid_data.target_edge_type),
#             (test_data.target_edge_index, test_data.target_edge_type),
#         ]:
#             head_mask[heads[(tails == dst) & (types == rel)]] = False

#         head = torch.arange(dataset.num_nodes)[head_mask]
#         head = torch.cat([torch.tensor([src]), head])
#         tail = torch.full_like(head, fill_value=dst)
#         eval_edge_index = torch.stack([head, tail], dim=0)
#         eval_edge_type = torch.full_like(head, fill_value=rel)

#         out = model.decode(z[eval_edge_index[0]], z[eval_edge_index[1]])
#         rank = compute_rank(out)
#         ranks.append(rank)

#     mrr = (1. / torch.tensor(ranks, dtype=torch.float)).mean()
#     hits1 = (torch.tensor(ranks) <= 1).to(torch.float32).mean()
#     hits3 = (torch.tensor(ranks) <= 3).to(torch.float32).mean()
#     hits10 = (torch.tensor(ranks) <= 10).to(torch.float32).mean()
    
#     return mrr, hits1, hits3, hits10



#%%
torch.manual_seed(seed)
torch_geometric.seed_everything(seed)

emb_dim = 16

model = GAE(
    RGCNet(num_node = dataset.num_nodes, num_relation = dataset.num_relations*2, emb_dim = emb_dim),
    MLPLinkPredictor(emb_dim, 1)
).to(device)
optimizer = optim. Adam(model.parameters(), lr = 0.001)


train_loss = []
for epoch in range(1, 101):
    loss = train(model, optimizer, train_data)
    train_loss.append(loss)
    
    
    valid_mrr, valid_hits1, valid_hits3, valid_hits10 = evaluation(model, valid_data)
    test_mrr, test_hits1, test_hits3, test_hits10 = evaluation(model, test_data)
    
    print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
    print(f'Val MRR: {valid_mrr:.3f}, Test MRR: {test_mrr:.3f}')
    print(f'Val Hits@1: {valid_hits1:.3f}, Test Hits@1: {test_hits1:.3f}')
    print(f'Val Hits@3: {valid_hits3:.3f}, Test Hits@3: {test_hits3:.3f}')
    print(f'Val Hits@10: {valid_hits10:.3f}, Test Hits@10: {test_hits10:.3f}')


#%%
