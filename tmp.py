# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py
# https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#hgtutorial
# https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
# https://www.kaggle.com/code/nadergo/link-prediction-on-a-heterogeneous-graph

#%%
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit


#%%
# raw 데이터 불러오기
chem_dis_path = 'dataset/raw/chem_dis.csv'
chem_dis_tmp = pd.read_csv(chem_dis_path, skiprows = list(range(27)) + [28])

gene_dis_path = 'dataset/raw/gene_dis.csv'
gene_dis_tmp = pd.read_csv(gene_dis_path, skiprows = list(range(27)) + [28])

chem_gene_path = 'dataset/raw/chem_gene.csv'
chem_gene_tmp = pd.read_csv(chem_gene_path, skiprows = list(range(27)) + [28])

chem_dis_tmp.DirectEvidence.value_counts()
gene_dis_tmp.DirectEvidence.value_counts()

# 각 데이터에서 사용할 column
chem_col = 'ChemicalID'
dis_col = 'DiseaseID'
gene_col = 'GeneID'


# mapping of unique chemical, disease, and gene
uniq_chem = pd.concat([chem_dis_tmp[chem_col], chem_gene_tmp[chem_col]]).unique()
chem_map = {name: i for i, name in enumerate(uniq_chem)}

uniq_dis = pd.concat([chem_dis_tmp[dis_col], gene_dis_tmp[dis_col]]).unique()
dis_map = {name: i for i, name in enumerate(uniq_dis)}

uniq_gene = pd.concat([gene_dis_tmp[gene_col], chem_gene_tmp[gene_col]]).unique()
gene_map = {str(name): i for i, name in enumerate(uniq_gene)}


# direct edge index between chem-disease
direct_chem_dis_idx = chem_dis_tmp.DirectEvidence == 'marker/mechanism'
direct_chem_dis = chem_dis_tmp[direct_chem_dis_idx][[chem_col, dis_col]]
direct_chem_dis = direct_chem_dis.drop_duplicates([chem_col, dis_col])
direct_chem_dis[chem_col] = direct_chem_dis[chem_col].apply(lambda x: chem_map[x])
direct_chem_dis[dis_col] = direct_chem_dis[dis_col].apply(lambda x: dis_map[x])


# curated edge index between chem-disease
curated_chem_dis_idx = chem_dis_tmp.DirectEvidence.isna()
curated_chem_dis = chem_dis_tmp[curated_chem_dis_idx][[chem_col, dis_col]]
curated_chem_dis = curated_chem_dis.drop_duplicates([chem_col, dis_col])
curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])


# direct evidence - therapeutic edge index between gene-disease
dir_thera_gene_dis_idx = gene_dis_tmp.DirectEvidence == 'therapeutic'
dir_thera_gene_dis = gene_dis_tmp[dir_thera_gene_dis_idx][[gene_col, dis_col]]
dir_thera_gene_dis = dir_thera_gene_dis.drop_duplicates([gene_col, dis_col])
dir_thera_gene_dis[gene_col] = dir_thera_gene_dis[gene_col].apply(lambda x: gene_map[str(x)])
dir_thera_gene_dis[dis_col] = dir_thera_gene_dis[dis_col].apply(lambda x: dis_map[x])


# direct evidence - marker edge index between gene-disease
direct_gene_dis_idx = gene_dis_tmp.DirectEvidence.notna()
dir_marker_gene_dis = gene_dis_tmp[direct_gene_dis_idx][[gene_col, dis_col, 'DirectEvidence']]
dir_marker_gene_dis_idx = dir_marker_gene_dis.DirectEvidence != 'therapeutic'
dir_marker_gene_dis = dir_marker_gene_dis[dir_marker_gene_dis_idx]
dir_marker_gene_dis = dir_marker_gene_dis[[gene_col, dis_col]].drop_duplicates([gene_col, dis_col])
dir_marker_gene_dis[gene_col] = dir_marker_gene_dis[gene_col].apply(lambda x: gene_map[str(x)])
dir_marker_gene_dis[dis_col] = dir_marker_gene_dis[dis_col].apply(lambda x: dis_map[x])


# curated edge index between gene-disease
curated_gene_dis_idx = gene_dis_tmp.DirectEvidence.isna()
curated_gene_dis = gene_dis_tmp[curated_gene_dis_idx][[gene_col, dis_col]]
curated_gene_dis = curated_gene_dis.drop_duplicates([gene_col, dis_col])
curated_gene_dis[gene_col] = curated_gene_dis[gene_col].apply(lambda x: gene_map[str(x)])
curated_gene_dis[dis_col] = curated_gene_dis[dis_col].apply(lambda x: dis_map[x])


# chem-gene
chem_gene = chem_gene_tmp.drop_duplicates([chem_col, gene_col])[[chem_col, gene_col]]
chem_gene[chem_col] = chem_gene[chem_col].apply(lambda x: chem_map[x])
chem_gene[gene_col] = chem_gene[gene_col].apply(lambda x: gene_map[str(x)])


#%%
# create heterogeneous graph
data = HeteroData()

data['chemical'].id = torch.tensor(list(chem_map.values()))
data['chemical'].x = torch.tensor(list(chem_map.values()))
data['disease'].id = torch.tensor(list(dis_map.values()))
data['disease'].x = torch.tensor(list(dis_map.values()))
data['gene'].id = torch.tensor(list(gene_map.values()))
data['gene'].x = torch.tensor(list(gene_map.values()))

data['chemical', 'cause', 'disease'].edge_index = torch.from_numpy(direct_chem_dis.values.T).to(torch.long)
data['chemical', 'relate', 'disease'].edge_index = torch.from_numpy(curated_chem_dis.values.T).to(torch.long)
# data['gene', 'cause', 'disease'].edge_index = torch.from_numpy(direct_gene_dis.values.T).to(torch.long)
data['gene', 'direct_therapeutic', 'disease'].edge_index = torch.from_numpy(dir_thera_gene_dis.values.T).to(torch.long)
data['gene', 'direct_marker', 'disease'].edge_index = torch.from_numpy(dir_marker_gene_dis.values.T).to(torch.long)
data['gene', 'relate', 'disease'].edge_index = torch.from_numpy(curated_gene_dis.values.T).to(torch.long)
data['chemical', 'relate', 'gene'].edge_index = torch.from_numpy(chem_gene.values.T).to(torch.long)

data = ToUndirected()(data)

del data['disease', 'rev_cause', 'chemical'].edge_label
del data['disease', 'rev_relate', 'chemical'].edge_label
del data['disease', 'rev_direct_therapeutic', 'gene'].edge_label
del data['disease', 'rev_direct_marker', 'gene'].edge_label
del data['disease', 'rev_relate', 'gene'].edge_label
del data['gene', 'rev_relate', 'chemical'].edge_label

data_split = RandomLinkSplit(
    num_val = 0.1, 
    num_test = 0.01,
    # num_test = 0.1,
    is_undirected = True,
    disjoint_train_ratio = 0.3,
    neg_sampling_ratio = 1.0,
    add_negative_train_samples = True,
    edge_types = ('chemical', 'cause', 'disease'),
    rev_edge_types = ('disease', 'rev_cause', 'chemical')
)

train_data, valid_data, test_data = data_split(data)


#%%
num_chem = len(data['chemical'].id)
num_disease = len(data['disease'].id)
num_gene = len(data['gene'].id)


class NodeEmbedding(nn.Module):
    def __init__(self, hidden_channels):
        super(NodeEmbedding, self).__init__()
        
        self.chem_emb = nn.Embedding(num_chem, hidden_channels)
        self.disease_emb = nn.Embedding(num_disease, hidden_channels)
        self.gene_emb = nn.Embedding(num_gene, hidden_channels)
        
    def forward(self, chem_id, dis_id, gene_id):
        x = {
            'chemical': self.chem_emb(chem_id),
            'disease': self.disease_emb(dis_id),
            'gene': self.gene_emb(gene_id)
        }
        return x
        

class GraphSage(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super(GraphSage, self).__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p = self.dropout, training = self.training)
        x = self.convs[-1](x, edge_index)
        
        return x


class LinkPredictor(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        
        self.lins = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
    
    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p = self.dropout, training = self.training)
        x = self.lins[-1](x)
        
        return torch.sigmoid(x)


# Loader를 사용하는 경우
# class Model(nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_layers, dropout, metadata):
#         super(Model, self).__init__()
        
#         self.chem_emb = nn.Embedding(num_chem, hidden_channels)
#         self.disease_emb = nn.Embedding(num_disease, hidden_channels)
#         self.gene_emb = nn.Embedding(num_gene, hidden_channels)
        
#         self.gnn = GraphSage(hidden_channels, num_layers, dropout)
#         self.gnn = to_hetero(self.gnn, metadata = metadata)
#         self.predictor = LinkPredictor(hidden_channels, out_channels, num_layer, dropout)
    
#     def forward(self, data):
#         x_dict = {
#             'chemical': self.chem_emb(data['chemical'].id),
#             'disease': self.disease_emb(data['disease'].id),
#             'gene': self.gene_emb(data['gene'].id)
#         }
#         x_dict = self.gnn(x_dict, data.edge_index_dict)
        
#         chem_idx, dis_idx = data['chemical', 'cause', 'disease'].edge_label_index
#         pred = self.predictor(x_dict['chemical'][chem_idx], x_dict['disease'][dis_idx])
        
#         return pred
     

#%%
def train(models, optimizer, criterion, data, batch_size):
    node_embed, gnn, predictor = models
    
    node_embed.train()
    gnn.train()
    predictor.train()
    
    total_loss, total_examples = 0, 0
    for perm in DataLoader(
        range(data.edge_label_index_dict[('chemical', 'cause', 'disease')].size(1)), 
        batch_size = batch_size,
        shuffle = True):
        
        optimizer.zero_grad()
        
        node_embedding = node_embed(data['chemical'].id, data['disease'].id, data['gene'].id)
        node_embedding = gnn(node_embedding, data.edge_index_dict)
        
        src, dst = data.edge_label_index_dict[('chemical', 'cause', 'disease')][:, perm]
        
        pred = predictor(node_embedding['chemical'][src], node_embedding['disease'][dst])
        target = data[('chemical', 'cause', 'disease')].edge_label[perm]
        
        loss = criterion(pred, target.view(pred.shape))
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.detach()) * pred.numel()
        total_examples += pred.numel()
    
    return total_loss/total_examples


@torch.no_grad()
def evaluation():



#%%
hidden_channels = 256
out_channels = 1
num_layer = 3
dropout = 0

node_embed = NodeEmbedding(hidden_channels)
gnn = GraphSage(hidden_channels, num_layer, dropout)
gnn = to_hetero(gnn, data.metadata())
predictor = LinkPredictor(hidden_channels, out_channels, num_layer, dropout)

criterion = nn.BCELoss()
models = (node_embed, gnn, predictor)

param_groups = []
param_groups.append({'params': node_embed.parameters()})
param_groups.append({'params': gnn.parameters()})
param_groups.append({'params': predictor.parameters()})
optimizer = optim.Adam(param_groups, lr = 0.001)

train(models, optimizer, criterion, test_data, batch_size = 64)







#%%
# Link 의 개수에 대한 분석  
# (inference score 가 붙은것 개수, 
# direct cause 가 붙은것 개수, C-D  에 대해서, 혹시 다른 연결관계 알 수 있는 것 있는지 확인) , 
# 연결 가능한 링크중에서 몇 개가 연결되어 있는지 비율

# - Direct cause 링크가 많이 붙어 있는 Disease
# - Inference score 의 합, 최대값 등이 큰 Disease 

import torch_geometric
import matplotlib.pyplot as plt



# direct cause
dir_row, dir_col = data[('chemical', 'cause', 'disease')].edge_index
dir_chem_degree = torch_geometric.utils.degree(dir_row)
dir_chem_topk_degree_idx = torch.argsort(dir_chem_degree)[-1] # top k chemical id

dir_chem_degree[dir_chem_topk_degree_idx]

list(chem_map.keys())[dir_chem_topk_degree_idx]
# [list(chem_map.keys())[i] for i in dir_chem_topk_degree_idx]

chem_dis_tmp[chem_dis_tmp.ChemicalID == 'D003042']
direct_chem_dis[direct_chem_dis.ChemicalID == dir_chem_topk_degree_idx.item()]

#
dir_dis_degree = torch_geometric.utils.degree(dir_col)
dir_dis_topk_degree_idx = torch.argsort(dir_dis_degree)[-1] # top k chemical id

dir_dis_degree[dir_dis_topk_degree_idx]

list(dis_map.keys())[dir_dis_topk_degree_idx]
# [list(dis_map.keys())[i] for i in dir_dis_topk_degree_idx]

chem_dis_tmp[chem_dis_tmp.DiseaseID == 'MESH:D056486']


# inferred relation
rel_row, rel_col = data[('chemical', 'relate', 'disease')].edge_index
rel_chem_degree = torch_geometric.utils.degree(rel_row)
rel_chem_topk_degree_idx = torch.argsort(rel_chem_degree)[-1]

rel_chem_degree[rel_chem_topk_degree_idx]
list(chem_map.keys())[rel_chem_topk_degree_idx]
# [list(chem_map.keys())[i] for i in rel_chem_topk_degree_idx]

chem_dis_tmp[chem_dis_tmp.ChemicalID == 'C006780']

#
rel_dis_degree = torch_geometric.utils.degree(rel_col)
rel_dis_topk_degree_idx = torch.argsort(rel_dis_degree)[-1] # top k chemical id

rel_dis_degree[rel_dis_topk_degree_idx]

list(dis_map.keys())[rel_dis_topk_degree_idx]
# [list(dis_map.keys())[i] for i in dir_dis_topk_degree_idx]

chem_dis_tmp[chem_dis_tmp.DiseaseID == 'MESH:D001943']


a = chem_dis_tmp.groupby(['DiseaseID'])['InferenceScore'].sum()
a[a == a.max()]
chem_dis_tmp[chem_dis_tmp.DiseaseID == 'MESH:D008106']
chem_dis_tmp[chem_dis_tmp.DiseaseID == 'MESH:D008106'].InferenceScore.sum()

b = chem_dis_tmp.groupby(['DiseaseID'])['InferenceScore'].max()
b[b == b.max()]
chem_dis_tmp[chem_dis_tmp.DiseaseID == 'MESH:D008106']


# %%
chem_dis_tmp.DirectEvidence.unique()

chem_dis_tmp.InferenceScore.max()
chem_dis_tmp.InferenceScore.min()
chem_dis_tmp.InferenceScore.median()
chem_dis_tmp.InferenceScore.mean()
chem_dis_tmp.InferenceScore.std()

plt.style.use('bmh')

fig = plt.figure(figsize = (14, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(chem_dis_tmp.InferenceScore, bins = 100, density = True, color = 'maroon')
ax1.set_xlabel('Inference Score')

ax2.hist(chem_dis_tmp.InferenceScore[chem_dis_tmp.InferenceScore <= 100], bins = 100, density = True, color = 'maroon')
ax2.set_xlabel('Inference Score')

plt.show()
plt.close()


chem_dis_tmp.DirectEvidence.value_counts()
a = chem_dis_tmp[chem_dis_tmp.DirectEvidence.notna()]
a[a.duplicated(['ChemicalID', 'DiseaseID'])]

a[a.ChemicalID == 'C020549']

chem_dis_tmp[chem_dis_tmp.DirectEvidence.notna()].drop_duplicates(['ChemicalID', 'DiseaseID'])
chem_dis_tmp[chem_dis_tmp.DirectEvidence.notna()].InferenceScore.notna().sum()