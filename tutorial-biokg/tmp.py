# https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/biokg/dataloader.py
# https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/dataloader.py

''' Negative Sampling '''

import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import LinkPropPredDataset, Evaluator

from dataloader import TrainDataset, BidirectionalOneShotIterator


dataset_name = 'ogbl-biokg'
dataset = LinkPropPredDataset(name = dataset_name)
dataset[0]
dataset[0].keys()
dataset[0]['edge_index_dict'][('drug', 'drug-sideeffect', 'sideeffect')].max()

num_edges = 0
for edge in dataset[0]['edge_index_dict'].values():
    num_edges += edge.shape[1]



# 모든 entity의 id는 0부터 시작
max_list = [i.max() for i in dataset[0]['edge_index_dict'].values()]
max(max_list)
dataset[0]['num_nodes_dict']



split_edge = dataset.get_edge_split()
train_triples, valid_triples, test_triples = split_edge['train'], split_edge['valid'], split_edge['test']
len(train_triples['head'])/5088434
len(valid_triples['head'])/5088434
len(test_triples['head'])/5088434

train_triples.keys()
train_triples['head_type']
train_triples['head']
train_triples['relation']
train_triples['tail_type']
train_triples['tail']

len(valid_triples['head'])
len(valid_triples['tail'])
len(test_triples['head'])

entity_dict = dict()
cur_idx = 0
for key in dataset[0]['num_nodes_dict']:
    entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
    cur_idx += dataset[0]['num_nodes_dict'][key]
nentity = sum(dataset[0]['num_nodes_dict'].values())
nrelation = int(max(train_triples['relation'])) + 1

train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
for i in range(len(train_triples['head'])):
    head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
    head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
    train_count[(head, relation, head_type)] += 1
    train_count[(tail, -relation-1, tail_type)] += 1
    train_true_head[(relation, tail)].append(head)
    train_true_tail[(head, relation)].append(tail)


# train dataset은 randomly negative sampling: head-batch mode
# head-batch는 head가 바뀌는거 / tail-batch는 tail이 바뀌는거
train_dataset = TrainDataset(
    train_triples,
    nentity,
    nrelation,
    128,
    'tail-batch',
    train_count,
    train_true_head,
    train_true_tail,
    entity_dict
)
p, n, w, m = train_dataset.__getitem__(0)

torch.manual_seed(0)
train_dataloader_head = DataLoader(
    TrainDataset(
        train_triples,
        nentity,
        nrelation,
        128,
        'head-batch',
        train_count,
        train_true_head,
        train_true_tail,
        entity_dict
    ),
    batch_size = 1024,
    shuffle = True,
    collate_fn = TrainDataset.collate_fn
)

for b1 in train_dataloader_head:
    break


torch.manual_seed(0)
train_dataloader_tail = DataLoader(
    TrainDataset(
        train_triples,
        nentity,
        nrelation,
        128,
        'tail-batch',
        train_count,
        train_true_head,
        train_true_tail,
        entity_dict
    ),
    batch_size = 1024,
    shuffle = True,
    collate_fn = TrainDataset.collate_fn
)

for b2 in train_dataloader_tail:
    break


torch.manual_seed(0)
train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

b = next(train_iterator)
b[0].shape


# validation/test dataset은 negative sample을 포함하고 있음
valid_triples.keys()
np.unique(valid_triples['head_type'])
valid_triples['head'].max()
valid_triples['head'].shape
valid_triples['head_neg'].max()
valid_triples['head_neg'].shape

test_triples.keys()
test_triples['head'].max()
test_triples['head'].shape
test_triples['head_neg'].max()
test_triples['head_neg'].shape
test_triples['tail'].max()
test_triples['tail'].shape
test_triples['tail_neg'].max()
test_triples['tail_neg'].shape
test_triples['relation']





#%%
evaluator = Evaluator(name = dataset_name)

for i in tqdm(range(len(train_triples['head']))):
    head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
    head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
    train_count[(head, relation, head_type)] += 1
    train_count[(tail, -relation-1, tail_type)] += 1
    train_true_head[(relation, tail)].append(head)
    train_true_tail[(head, relation)].append(tail)

kge_model = KGEModel(
    model_name = 'TransE',
    nentity = nentity,
    nrelation = nrelation,
    hidden_dim = 500,
    gamma = 12.0,
    double_entity_embedding = True,
    double_relation_embedding = True,
    evaluator = evaluator
)

train_dataloader_head = DataLoader(
    TrainDataset(
        train_triples,
        nentity,
        nrelation,
        128,
        'head-batch',
        train_count,
        train_true_head,
        train_true_tail,
        entity_dict
    ),
    batch_size = 1024,
    shuffle = True,
    collate_fn = TrainDataset.collate_fn
)

train_dataloader_tail = DataLoader(
    TrainDataset(
        train_triples,
        nentity,
        nrelation,
        128,
        'tail-batch',
        train_count,
        train_true_head,
        train_true_tail,
        entity_dict
    ),
    batch_size = 1024,
    shuffle = True,
    collate_fn = TrainDataset.collate_fn
)

train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

current_learning_rate = 0.0001
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, kge_model.parameters()),
    lr = current_learning_rate
)

init_step = 0
step = init_step
max_step = 100

kge_model.train_step(kge_model, optimizer, train_iterator, args)



#%%
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode