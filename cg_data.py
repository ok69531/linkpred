#%%
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from itertools import repeat
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData


#%%
'''
    Chemical-Gene Interaction data
    - organism이 nan인 데이터 제거
    - geneforms가 nan인 데이터 제거

    1. geneforms을 구분하지 않고 그래프 생성
    2. gene entity를 geneforms로 세분화하여 그래프 생성. 
       이때 하나의 chemical이 여러가지 geneforms를 갖는 경우 해당 chemical은 여러 gene entity에 연결
'''

chem_gene_path = 'dataset/raw/chem_gene.csv'
chem_gene_tmp = pd.read_csv(chem_gene_path, skiprows = list(range(27)) + [28])

# 각 데이터에서 사용할 column
chem_col = 'ChemicalID'
gene_col = 'GeneID'


# organism이 명시되어 있지 않은 데이터는 제거
org_na_idx = chem_gene_tmp['Organism'].isna()
geneform_na_idx = chem_gene_tmp['GeneForms'].isna()
chem_gene = chem_gene_tmp[~(org_na_idx|geneform_na_idx)]

# remove duplicated chem-gene pair
chem_gene = chem_gene[[chem_col, gene_col]].drop_duplicates()


#%%
''' Heterogeneous Graph '''
# mapping of unique chemical, disease, and gene
uniq_chem = chem_gene[chem_col].unique()
chem_map = {name: i for i, name in enumerate(uniq_chem)}

uniq_gene = chem_gene[gene_col].unique()
gene_map = {name: i for i, name in enumerate(uniq_gene)}

edge_type_map = {
    'chem_inferred_gene': 0
}

# mapping the chemical and disease id
inferred_chem_gene = chem_gene.copy()
inferred_chem_gene[chem_col] = inferred_chem_gene[chem_col].apply(lambda x: chem_map[x])
inferred_chem_gene[gene_col] = inferred_chem_gene[gene_col].apply(lambda x: gene_map[x])

# torch.save(chem_map, 'dataset/cg/chem_map')
# torch.save(gene_map, 'dataset/cg/simple_gene_map')
# torch.save(edge_type_map, 'dataset/cg/simple_rel_type_map')


#%%
data = Data()
data.num_nodes_dict = {
    'chemical': len(chem_map),
    'gene': len(gene_map)
}
data.edge_index_dict = {
    ('chemical', 'chem_inferred_gene', 'gene'): torch.from_numpy(inferred_chem_gene.values.T).to(torch.long)
}
data.edge_reltype = {
    rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
}
data.num_relations = len(data.edge_index_dict)


# torch.save(data, 'dataset/cg/simple_cg.pt')


#%%
# edge 별로 train mask
# 90/5/5로 분할?

seed = 42
random.seed(seed)

train_frac = 0.9
valid_frac = 0.05


all_idx = {
    k: list(range(v.shape[1])) for k, v in data.edge_index_dict.items()
}
for k, v in all_idx.items():
    random.shuffle(v)

train_idx = {
    k: all_idx[k][:int(train_frac * len(v))] for k, v in all_idx.items()
}
valid_idx = {
    k: all_idx[k][int(train_frac * len(v)) : int(train_frac * len(v)) + int(valid_frac * len(v))] for k, v in all_idx.items()
}
test_idx = {
    k: all_idx[k][int(train_frac * len(v)) + int(valid_frac * len(v)):] for k, v in all_idx.items()
}

rel_type = list(edge_type_map.keys())
# len(train_idx[('chemical', 'chem_inferred_gene', 'gene')]) + len(valid_idx[('chemical', 'chem_inferred_gene', 'gene')]) + len(test_idx[('chemical', 'chem_inferred_gene', 'gene')]) == data.edge_index_dict[('chemical', 'chem_inferred_gene', 'gene')].shape[1]


train_data = Data()
train_data.num_nodes_dict = {
    'chemical': len(chem_map),
    'gene': len(gene_map)
}
train_data.edge_index_dict = {
    k: v[:, train_idx[k]] for k, v in data.edge_index_dict.items()
}
train_data.edge_reltype = {
    k: v[train_idx[k]] for k, v in data.edge_reltype.items()
}
train_data.num_relations = len(data.edge_index_dict)

# torch.save(train_data, 'dataset/cg/simple_cg_train_wo_neg.pt')


valid_data = Data()
valid_data.num_nodes_dict = {
    'chemical': len(chem_map),
    'gene': len(gene_map)
}
valid_data.edge_index_dict = {
    k: v[:, valid_idx[k]] for k, v in data.edge_index_dict.items()
}
valid_data.edge_reltype = {
    k: v[valid_idx[k]] for k, v in data.edge_reltype.items()
}
valid_data.num_relations = len(data.edge_index_dict)

# dict_keys(['head_type', 'head', 'relation', 'tail_type', 'tail'])
# head_neg, tail_neg 추가
head_type = []
tail_type = []
relation = []
head = []
tail = []
for (h, r, t), e in valid_data.edge_index_dict.items():
    head_type.append(list(repeat(h, e.shape[1])))
    tail_type.append(list(repeat(t, e.shape[1])))
    relation.append(valid_data.edge_reltype[(h, r, t)].view(-1))
    head.append(e[0])
    tail.append(e[1])

valid_triples = {
    'head_type': list(itertools.chain(*head_type)),
    'head': torch.cat(head),
    'relation': torch.cat(relation),
    'tail_type': list(itertools.chain(*tail_type)),
    'tail': torch.cat(tail)
}



test_data = Data()
test_data.num_nodes_dict = {
    'chemical': len(chem_map),
    'gene': len(gene_map)
}
test_data.edge_index_dict = {
    k: v[:, test_idx[k]] for k, v in data.edge_index_dict.items()
}
test_data.edge_reltype = {
    k: v[test_idx[k]] for k, v in data.edge_reltype.items()
}
test_data.num_relations = len(data.edge_index_dict)


head_type = []
tail_type = []
relation = []
head = []
tail = []
for (h, r, t), e in test_data.edge_index_dict.items():
    head_type.append(list(repeat(h, e.shape[1])))
    tail_type.append(list(repeat(t, e.shape[1])))
    relation.append(test_data.edge_reltype[(h, r, t)].view(-1))
    head.append(e[0])
    tail.append(e[1])

test_triples = {
    'head_type': list(itertools.chain(*head_type)),
    'head': torch.cat(head),
    'relation': torch.cat(relation),
    'tail_type': list(itertools.chain(*tail_type)),
    'tail': torch.cat(tail)
}


#%%
class NegativeSampling(Dataset):
    def __init__(self, triples, negative_sample_size, mode, true_head, true_tail, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.true_head = true_head
        self.true_tail = true_tail
        self.entity_dict = entity_dict
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx].item(), self.triples['relation'][idx].item(), self.triples['tail'][idx].item()
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = [head, relation, tail]
        
        if self.mode == 'head-batch':
            # head-batch는 tail은 그대로 유지되며 head가 다양하게 바뀌는 경우
            # 즉, (h', r, t)를 생성하는 것
            # 따라서 랜덤하게 head를 생성하고, (h', r, t)가 실제 triplet에 포함되어 있는지를 확인
            negative_sample = torch.randint(0, self.entity_dict[head_type], (self.negative_sample_size+100,))
            negative_sample = torch.stack([i for i in negative_sample if i not in set(self.true_head[(relation, tail)])])[:self.negative_sample_size]
        elif self.mode == 'tail-batch':
            negative_sample = torch.randint(0, self.entity_dict[tail_type], (self.negative_sample_size+100,))
            negative_sample = torch.stack([i for i in negative_sample if i not in set(self.true_tail[(head, relation)])])[:self.negative_sample_size]
        else:
            raise
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode


#%%
torch.manual_seed(seed)

valid_true_head, valid_true_tail = defaultdict(list), defaultdict(list)
for i in tqdm(range(len(valid_triples['head']))):
    head, relation, tail = valid_triples['head'][i].item(), valid_triples['relation'][i].item(), valid_triples['tail'][i].item()
    head_type, tail_type = valid_triples['head_type'][i], valid_triples['tail_type'][i]
    valid_true_head[(relation, tail)].append(head)
    valid_true_tail[(head, relation)].append(tail)

valid_head_negative_sampling = NegativeSampling(valid_triples, 500, 'head-batch', valid_true_head, valid_true_tail, data.num_nodes_dict)
valid_head_negative = []
for i in tqdm(range(len(valid_triples['head']))):
    valid_head_negative.append(valid_head_negative_sampling.__getitem__(i)[1])

valid_tail_negative_sampling = NegativeSampling(valid_triples, 500, 'tail-batch', valid_true_head, valid_true_tail, data.num_nodes_dict)
valid_tail_negative = []
for i in tqdm(range(len(valid_triples['tail']))):
    valid_tail_negative.append(valid_tail_negative_sampling.__getitem__(i)[1])

valid_data['head_neg'] = torch.stack(valid_head_negative)
valid_data['tail_neg'] = torch.stack(valid_tail_negative)

# torch.save(valid_data, 'dataset/cg/simple_cg_valid.pt')


#%%
torch.manual_seed(seed)

test_true_head, test_true_tail = defaultdict(list), defaultdict(list)
for i in tqdm(range(len(test_triples['head']))):
    head, relation, tail = test_triples['head'][i].item(), test_triples['relation'][i].item(), test_triples['tail'][i].item()
    head_type, tail_type = test_triples['head_type'][i], test_triples['tail_type'][i]
    test_true_head[(relation, tail)].append(head)
    test_true_tail[(head, relation)].append(tail)

test_head_negative_sampling = NegativeSampling(test_triples, 500, 'head-batch', test_true_head, test_true_tail, data.num_nodes_dict)
test_head_negative = []
for i in tqdm(range(len(valid_triples['head']))):
    test_head_negative.append(test_head_negative_sampling.__getitem__(i)[1])

test_tail_negative_sampling = NegativeSampling(test_triples, 500, 'tail-batch', test_true_head, test_true_tail, data.num_nodes_dict)
test_tail_negative = []
for i in tqdm(range(len(test_triples['tail']))):
    test_tail_negative.append(test_tail_negative_sampling.__getitem__(i)[1])

test_data['head_neg'] = torch.stack(test_head_negative)
test_data['tail_neg'] = torch.stack(test_tail_negative)

# torch.save(test_data, 'dataset/cg/simple_cg_test.pt')


# %%
''' 2. gene entity를 geneforms로 세분화하여 그래프 생성  '''

chem_gene_path = 'dataset/raw/chem_gene.csv'
chem_gene_tmp = pd.read_csv(chem_gene_path, skiprows = list(range(27)) + [28])

# 각 데이터에서 사용할 column
chem_col = 'ChemicalID'
gene_col = 'GeneID'


# organism이 명시되어 있지 않은 데이터는 제거
org_na_idx = chem_gene_tmp['Organism'].isna()
geneform_na_idx = chem_gene_tmp['GeneForms'].isna()
chem_gene = chem_gene_tmp[~(org_na_idx|geneform_na_idx)]


# detecting multiple geneforms
split_geneform = chem_gene['GeneForms'].map(lambda x: x.split('|'))
uniq_geneform = set(itertools.chain(*split_geneform))

multi_geneform_idx = split_geneform.map(lambda x: len(x))

single_chem_gene = chem_gene[multi_geneform_idx == 1]
double_chem_gene = chem_gene[multi_geneform_idx == 2]
triple_chem_gene = chem_gene[multi_geneform_idx == 3]


# split dataframe per geneform
single_chem_gene['GeneForms'].value_counts()
geneform_df_dict = {g: chem_gene[chem_gene['GeneForms']==g] for g in uniq_geneform}


# insert data which have multiple geneforms to geneform_df 
double_chem_gene



# remove duplicated chem-gene pair
chem_gene = chem_gene[[chem_col, gene_col]].drop_duplicates()