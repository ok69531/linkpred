# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py
# https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#hgtutorial
# https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

#%%
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torch_geometric
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from torch_geometric.utils import k_hop_subgraph


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
''' read data '''
# raw 데이터 불러오기
chem_dis_path = 'dataset/raw/chem_dis.csv'
chem_dis_tmp = pd.read_csv(chem_dis_path, skiprows = list(range(27)) + [28])

gene_dis_path = 'dataset/raw/gene_dis.csv'
gene_dis_tmp = pd.read_csv(gene_dis_path, skiprows = list(range(27)) + [28])

chem_gene_path = 'dataset/raw/chem_gene.csv'
chem_gene_tmp = pd.read_csv(chem_gene_path, skiprows = list(range(27)) + [28])


# 각 데이터에서 사용할 column
chem_col = 'ChemicalID'
dis_col = 'DiseaseID'
gene_col = 'GeneID'


# target disease로 제한
target_dis_tmp = pd.read_excel('dataset/raw/CTD_disease_list.xlsx', sheet_name = ['target', 'hierarchy'])
target_dis_list = target_dis_tmp['target']
target_dis_list['MESH ID'] = 'MESH:'+target_dis_list['MESH ID']

target_chem_dis = chem_dis_tmp[chem_dis_tmp[dis_col].isin(target_dis_list['MESH ID'])]
target_gene_dis = gene_dis_tmp[gene_dis_tmp[dis_col].isin(target_dis_list['MESH ID'])]


#%%
# remove chem_dis_tmp and gen_dis_tmp
del chem_dis_tmp
del gene_dis_tmp


#%%
# mapping of unique chemical, disease, and gene
uniq_chem = pd.concat([target_chem_dis[chem_col], chem_gene_tmp[chem_col]]).unique()
chem_map = {name: i for i, name in enumerate(uniq_chem)}

uniq_gene = pd.concat([target_gene_dis[gene_col], chem_gene_tmp[gene_col]]).unique()
gene_map = {str(name): i for i, name in enumerate(uniq_gene)}

uniq_dis = pd.concat([target_chem_dis[dis_col], target_gene_dis[dis_col]]).unique()
dis_map = {name: i for i, name in enumerate(uniq_dis)}


# uniq_chem = pd.concat([target_chem_dis[chem_col], chem_gene_tmp[chem_col]]).unique()
# chem_map = {name: i for i, name in enumerate(uniq_chem)}

# uniq_gene = pd.concat([target_gene_dis[gene_col], chem_gene_tmp[gene_col]]).unique()
# gene_map = {str(name): i+len(chem_map) for i, name in enumerate(uniq_gene)}

# uniq_dis = pd.concat([target_chem_dis[dis_col], target_gene_dis[dis_col]]).unique()
# dis_map = {name: i+len(chem_map)+len(gene_map) for i, name in enumerate(uniq_dis)}


#%%
''' create edges '''
target_chem_dis.DirectEvidence.value_counts()
target_gene_dis.DirectEvidence.value_counts()

# direct edge index between chem-disease
direct_chem_dis_idx = target_chem_dis.DirectEvidence == 'marker/mechanism'
direct_chem_dis = target_chem_dis[direct_chem_dis_idx][[chem_col, dis_col]]
assert direct_chem_dis.duplicated(keep=False).sum() == 0

# inferenced relationship between chem-disease
curated_chem_dis_idx = target_chem_dis.DirectEvidence.isna()
curated_chem_dis = target_chem_dis[curated_chem_dis_idx][[chem_col, dis_col]]
curated_chem_dis = curated_chem_dis.drop_duplicates([chem_col, dis_col])

# drop the duplicated pair of chem-disease which have direct evidence and inference score
dup_chem_idx = curated_chem_dis[chem_col].isin(direct_chem_dis[chem_col])
dup_dis_idx = curated_chem_dis[dis_col].isin(direct_chem_dis[dis_col])
curated_chem_dis = curated_chem_dis[~(dup_chem_idx & dup_dis_idx)]


# direct edge index between gene-disease
direct_gene_dis_idx = target_gene_dis.DirectEvidence.isin(['marker/mechanism', 'marker/mechanism|therapeutic'])
direct_gene_dis = target_gene_dis[direct_gene_dis_idx][[gene_col, dis_col]]
assert direct_chem_dis.duplicated(keep=False).sum() == 0

# inferenced relationship between gene-disease
curated_gene_dis_idx = target_gene_dis.DirectEvidence.isna()
curated_gene_dis = target_gene_dis[curated_gene_dis_idx][[gene_col, dis_col]]
curated_gene_dis = curated_gene_dis.drop_duplicates([gene_col, dis_col])

# drop the duplicated pair of gene-disease which have direct evidence and inference score
dup_gene_idx = curated_gene_dis[gene_col].isin(direct_gene_dis[gene_col])
dup_dis_idx = curated_gene_dis[dis_col].isin(direct_gene_dis[dis_col])
curated_gene_dis = curated_gene_dis[~(dup_gene_idx & dup_dis_idx)]


# edge index between chem-gene
curated_chem_gene = chem_gene_tmp.drop_duplicates([chem_col, gene_col])[[chem_col, gene_col]]


# mapping the chemical and disease id
direct_chem_dis[chem_col] = direct_chem_dis[chem_col].apply(lambda x: chem_map[x])
direct_chem_dis[dis_col] = direct_chem_dis[dis_col].apply(lambda x: dis_map[x])

curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])

direct_gene_dis[gene_col] = direct_gene_dis[gene_col].apply(lambda x: gene_map[str(x)])
direct_gene_dis[dis_col] = direct_gene_dis[dis_col].apply(lambda x: dis_map[x])

curated_gene_dis[gene_col] = curated_gene_dis[gene_col].apply(lambda x: gene_map[str(x)])
curated_gene_dis[dis_col] = curated_gene_dis[dis_col].apply(lambda x: dis_map[x])

curated_chem_gene[chem_col] = curated_chem_gene[chem_col].apply(lambda x: chem_map[x])
curated_chem_gene[gene_col] = curated_chem_gene[gene_col].apply(lambda x: gene_map[str(x)])


#%%
# create heterogeneous graph
hetero_data = HeteroData()

hetero_data['chemical'].id = torch.tensor(list(chem_map.values()))
hetero_data['chemical'].x = torch.tensor(list(chem_map.values()))
hetero_data['disease'].id = torch.tensor(list(dis_map.values()))
hetero_data['disease'].x = torch.tensor(list(dis_map.values()))
hetero_data['gene'].id = torch.tensor(list(gene_map.values()))
hetero_data['gene'].x = torch.tensor(list(gene_map.values()))

hetero_data['chemical', 'cause', 'disease'].edge_index = torch.from_numpy(direct_chem_dis.values.T).to(torch.long)
hetero_data['chemical', 'relate', 'disease'].edge_index = torch.from_numpy(curated_chem_dis.values.T).to(torch.long)
hetero_data['disease', 'rev_relate', 'chemical'].edge_index = torch.from_numpy(curated_chem_dis.values.T[[1, 0], :]).to(torch.long)
hetero_data['gene', 'cause', 'disease'].edge_index = torch.from_numpy(direct_gene_dis.values.T).to(torch.long)
hetero_data['gene', 'relate', 'disease'].edge_index = torch.from_numpy(curated_gene_dis.values.T).to(torch.long)
hetero_data['disease', 'rev_relate', 'gene'].edge_index = torch.from_numpy(curated_gene_dis.values.T[[1, 0], :]).to(torch.long)
hetero_data['chemical', 'relate', 'gene'].edge_index = torch.from_numpy(curated_chem_gene.values.T).to(torch.long)
hetero_data['gene', 'rev_relate', 'chemical'].edge_index = torch.from_numpy(curated_chem_gene.values.T[[1, 0], :]).to(torch.long)

# torch.save(hetero_data, 'dataset/ctd_graph_tmp.pt')


#%%
# hetero_data = HeteroData()

# hetero_data['chemical'].nx_id = torch.tensor(list(chem_map.values()))
# hetero_data['gene'].nx_id = torch.tensor(list(gene_map.values()))
# hetero_data['disease'].nx_id = torch.tensor(list(dis_map.values()))

# hetero_data['chemical', 'chem_cause_dis', 'disease'].edge_index = torch.from_numpy(direct_chem_dis.values.T).to(torch.long)
# hetero_data['chemical', 'chem_relate_dis', 'disease'].edge_index = torch.from_numpy(curated_chem_dis.values.T).to(torch.long)
# hetero_data['disease', 'dis_rev_relate_chem', 'chemical'].edge_index = torch.from_numpy(curated_chem_dis.values.T[[1, 0], :]).to(torch.long)
# hetero_data['gene', 'gene_cause_dis', 'disease'].edge_index = torch.from_numpy(direct_gene_dis.values.T).to(torch.long)
# hetero_data['gene', 'gene_relate_dis', 'disease'].edge_index = torch.from_numpy(curated_gene_dis.values.T).to(torch.long)
# hetero_data['disease', 'dis_rev_relate_gene', 'gene'].edge_index = torch.from_numpy(curated_gene_dis.values.T[[1, 0], :]).to(torch.long)
# hetero_data['chemical', 'chem_relate_gene', 'gene'].edge_index = torch.from_numpy(curated_chem_gene.values.T).to(torch.long)
# hetero_data['gene', 'gene_rev_relate_chem', 'chemical'].edge_index = torch.from_numpy(curated_chem_gene.values.T[[1, 0], :]).to(torch.long)


# hetero_data['chemical', 'chem_cause_dis', 'disease'].count = torch.ones(len(direct_chem_dis.values)).to(torch.long)
# hetero_data['chemical', 'chem_relate_dis', 'disease'].count = torch.ones(len(curated_chem_dis.values)).to(torch.long)
# hetero_data['disease', 'dis_rev_relate_chem', 'chemical'].count = torch.ones(len(curated_chem_dis.values)).to(torch.long)
# hetero_data['gene', 'gene_cause_dis', 'disease'].count = torch.ones(len(direct_gene_dis.values)).to(torch.long)
# hetero_data['gene', 'gene_relate_dis', 'disease'].count = torch.ones(len(curated_gene_dis.values)).to(torch.long)
# hetero_data['disease', 'dis_rev_relate_gene', 'gene'].count = torch.ones(len(curated_gene_dis.values)).to(torch.long)
# hetero_data['chemical', 'chem_relate_gene', 'gene'].count = torch.ones(len(curated_chem_gene.values)).to(torch.long)
# hetero_data['gene', 'gene_rev_relate_chem', 'chemical'].count = torch.ones(len(curated_chem_gene.values)).to(torch.long)


# #%%
# def to_dgl(data):
#     r"""Converts a :class:`torch_geometric.data.Data` or
#     :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
#     object.

#     Args:
#         data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
#             The data object.

#     Example:
#         >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
#         >>> x = torch.randn(5, 3)
#         >>> edge_attr = torch.randn(6, 2)
#         >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
#         >>> g = to_dgl(data)
#         >>> g
#         Graph(num_nodes=5, num_edges=6,
#             ndata_schemes={'x': Scheme(shape=(3,))}
#             edata_schemes={'edge_attr': Scheme(shape=(2, ))})

#         >>> data = HeteroData()
#         >>> data['paper'].x = torch.randn(5, 3)
#         >>> data['author'].x = torch.ones(5, 3)
#         >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
#         >>> data['author', 'cites', 'paper'].edge_index = edge_index
#         >>> g = to_dgl(data)
#         >>> g
#         Graph(num_nodes={'author': 5, 'paper': 5},
#             num_edges={('author', 'cites', 'paper'): 5},
#             metagraph=[('author', 'paper', 'cites')])
#     """
#     import dgl

#     from torch_geometric.data import Data, HeteroData

#     if isinstance(data, Data):
#         if data.edge_index is not None:
#             row, col = data.edge_index
#         else:
#             row, col, _ = data.adj_t.t().coo()

#         g = dgl.graph((row, col))

#         for attr in data.node_attrs():
#             g.ndata[attr] = data[attr]
#         for attr in data.edge_attrs():
#             if attr in ['edge_index', 'adj_t']:
#                 continue
#             g.edata[attr] = data[attr]

#         return g

#     if isinstance(data, HeteroData):
#         data_dict = {}
#         for edge_type, edge_store in data.edge_items():
#             if edge_store.get('edge_index') is not None:
#                 row, col = edge_store.edge_index
#             else:
#                 row, col, _ = edge_store['adj_t'].t().coo()

#             data_dict[edge_type] = (row, col)

#         g = dgl.heterograph(data_dict)

#         for node_type, node_store in data.node_items():
#             for attr, value in node_store.items():
#                 g.nodes[node_type].data[attr] = value

#         for edge_type, edge_store in data.edge_items():
#             for attr, value in edge_store.items():
#                 if attr in ['edge_index', 'adj_t']:
#                     continue
#                 g.edges[edge_type].data[attr] = value

#         return g

#     raise ValueError(f"Invalid data type (got '{type(data)}')")


# #%%
# import dgl
# a = to_dgl(hetero_data)

# a.nodes['chemical']
# a.nodes['gene']
# a.nodes['disease']
# a.edges[('chemical', 'chem_cause_dis', 'disease')]
# dgl.save_graphs('page-link/datasets/ctd', a)


#%%
def convert_hetero_object_to_RGCN(heteroData, edge_types):
    dataRGCN = Data()
    datax  =  torch.arange(len(heteroData['chemical'].x) +len(heteroData['gene'].x) + len(heteroData['disease'].x))
    data_edge_index = torch.tensor([]).reshape(2,0)
    data_edge_types = torch.tensor([])
    # edge_types = []   
    for _, (key,v) in enumerate(heteroData.edge_index_dict.copy().items()):
        ourV = torch.tensor(v.cpu().numpy())
        if key[0] == "disease":
            ourV[0,:] = ourV[0]+len(heteroData["chemical"].x) + len(heteroData["gene"].x) 
        if key[0] == "gene":
            ourV[0,:] = ourV[0]+len(heteroData["chemical"].x)
        
        if key[1] == "disease":
            ourV[1,:] = ourV[1]+len(heteroData["chemical"].x) + len(heteroData["gene"].x)
        if key[1] == "gene":
            ourV[1,:] = ourV[1]+len(heteroData["disease"].x)
            
        data_edge_index = torch.cat([data_edge_index,ourV],dim = -1)
        data_edge_types = torch.cat([data_edge_types, torch.zeros(len(ourV[0])) + edge_types.index(key)])
    
    dataRGCN.edge_index = data_edge_index.to(torch.long)
    dataRGCN.x = datax.to(torch.long)
    
    edge_label_index =torch.tensor( heteroData["chemical","cause","disease"].edge_label_index.cpu().numpy())
    edge_label_index[1,:] += (len(heteroData["chemical"].x) + len(heteroData["gene"].x)  )
    edge_label = torch.tensor(heteroData["chemical","cause","disease"].edge_label.cpu().numpy())
    target_label_type = torch.zeros(len(edge_label)) + heteroData.edge_types.index(("chemical","cause","disease"))
    dataRGCN.edge_label_index = edge_label_index
    dataRGCN.edge_label = edge_label
    dataRGCN.edge_type = data_edge_types.to(torch.long)
    dataRGCN.label_edge_type = target_label_type.to(torch.long)
    return  dataRGCN


#%%
class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, num_layers, emb_dim = 32, num_blocks = 5, dropout = 0):
        super(RGCNEncoder,self).__init__()

        self.dropout = dropout
        self.node_embed = nn.Embedding(num_nodes, emb_dim)
    
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = RGCNConv(emb_dim, emb_dim, num_relations, num_blocks) 
            self.convs.append(conv)
        
        
    def forward(self, x, edge_index, edge_type):
        x = self.node_embed(x)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p = self.dropout, training = self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        
        return x


class DistMult(nn.Module):
    def __init__(self, num_relations, emb_dim):
        super(DistMult, self).__init__()
        
        self.relation_embedding = nn.Embedding(num_relations, emb_dim)

    def forward(self, x_i, x_j, edge_type):
        relation_embedding = self.relation_embedding(edge_type)
        score = torch.sum(x_i * relation_embedding * x_j, dim = -1)
        return score


class TransE(nn.Module):
    def __init__(self, num_relations, emb_dim):
        super(TransE, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, emb_dim)

    def forward(self, x_i, x_j, edge_type):
        relation_embedding = self.relation_embedding(edge_type)
        score = torch.norm(x_i + relation_embedding - x_j, p = 1, dim = -1)
        
        return score


class MultiLayerPerceptron(nn.Module):
    def __init__(self, emb_dim, out_dim, num_layers, dropout):
        super(MultiLayerPerceptron, self).__init__()
        
        self.lins = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(emb_dim, emb_dim))
        self.lins.append(nn.Linear(emb_dim, out_dim))
        
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
        score = self.lins[-1](x)
        
        return torch.sigmoid(score)


class LinkPredictor(nn.Module):
    def __init__(self, model_name, num_relations, emb_dim, out_dim, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        
        if model_name not in ['transe', 'distmult', 'mlp']:
            raise ValueError(f'model {model_name} not supported')
        
        if model_name == 'transe':
            self.decoder = TransE(num_relations, emb_dim)
        elif model_name == 'distmult':
            self.decoder = DistMult(num_relations, emb_dim)
        elif model_name == 'mlp':
            self.decoder = MultiLayerPerceptron(emb_dim, out_dim, num_layers, dropout)
        
        self.model_name = model_name
        
    def forward(self, x_i, x_j, edge_type = None):
        if self.model_name == 'mlp':
            score = self.decoder(x_i, x_j)
        else:
            score = self.decoder(x_i, x_j, edge_type)
        
        return score


def nsloss(pred, target):
    pos_idx = target == 1
    neg_idx = target == 0
    
    positive_score = nn.functional.logsigmoid(pred[pos_idx])
    positive_score = -positive_score.mean()
    negative_score = nn.functional.logsigmoid(-pred[neg_idx])
    negative_score = -negative_score.mean()
    
    loss = (positive_score + negative_score) / 2
    
    return loss


# #%%
# def train(models, optimizer, data):
#     rgcn, predictor = models
    
#     rgcn.train()
#     predictor.train()
        
#     optimizer.zero_grad()
    
#     node_embedding = rgcn(data.x, data.edge_index, data.edge_type) 
#     src, dst = data.edge_label_index
    
#     pred = predictor(node_embedding[src], node_embedding[dst], data.label_edge_type)
#     target = data.edge_label
    
#     loss = nsloss(pred, target)
#     loss.backward()
#     optimizer.step()
    
#     return loss


# @torch.no_grad()
# def evaluation(models, data):
#     rgcn, predictor = models
    
#     rgcn.eval()
#     predictor.eval()
    
#     src, dst = data.edge_label_index
    
#     node_embedding = rgcn(data.x, data.edge_index, data.edge_type) 
#     pred = predictor(node_embedding[src], node_embedding[dst], data.label_edge_type)
#     target = data.edge_label
    
#     loss = nsloss(pred, target)
#     auc = roc_auc_score(target.cpu().numpy(), pred.detach().cpu().numpy())
    
#     return loss, auc


# #%%
# seed = 0

# torch.manual_seed(seed)
# torch_geometric.seed_everything(seed)

# data_split = RandomLinkSplit(
#     num_val = 0.1, 
#     num_test = 0.1,
#     is_undirected = False,
#     disjoint_train_ratio = 0.3,
#     neg_sampling_ratio = 1.0,
#     add_negative_train_samples = True,
#     edge_types = ('chemical', 'cause', 'disease')
# )

# train_data, valid_data, test_data = data_split(hetero_data)

# # rgcn_train = train_data.to_homogeneous()
# # rgcn_valid = valid_data.to_homogeneous()
# # rgcn_test = test_data.to_homogeneous()
# rgcn_train = convert_hetero_object_to_RGCN(train_data, hetero_data.edge_types)
# rgcn_valid = convert_hetero_object_to_RGCN(valid_data, hetero_data.edge_types)
# rgcn_test = convert_hetero_object_to_RGCN(test_data, hetero_data.edge_types)


# num_layer = 3
# out_dim = 1
# dropout = 0
# emb_dim = 300
# num_blocks = 5
# num_nodes = len(rgcn_train.x)
# num_relations = len(hetero_data.edge_index_dict)


# model_name = 'distmult'
# rgcn = RGCNEncoder(num_nodes, num_relations, num_layer, emb_dim, num_blocks, dropout)
# predictor = LinkPredictor(model_name, num_relations, emb_dim, out_dim, num_layer, dropout)

# models = (rgcn, predictor)

# param_groups = []
# param_groups.append({'params': rgcn.parameters()})
# param_groups.append({'params': predictor.parameters()})
# optimizer = optim.Adam(param_groups, lr = 0.001)


# epochs = 100
# train_loss = []
# best_val_auc, final_test_auc = 0, 0

# for epoch in range(1, epochs+1):
#     _train_loss = train(models, optimizer, rgcn_train)
#     train_loss.append(_train_loss)
    
#     val_loss, val_auc = evaluation(models, rgcn_valid)
#     test_loss, test_auc = evaluation(models, rgcn_test)
    
#     if val_auc > best_val_auc:
#         best_val_auc = val_auc
#         final_test_auc = test_auc
        
#         rgcn_params = deepcopy(models[0].state_dict())
#         link_perd_params = deepcopy(models[1].state_dict())
    
#     print(f'=== epoch {epoch} ===')
#     print(f'train loss: {_train_loss:.3f}, validation loss: {val_loss:.3f}, test loss: {test_loss:.3f}')
#     print(f'validation auc: {val_auc:.3f}, test auc: {test_auc:.3f}')


# #%%
# plt.figure(figsize = (7, 5))
# plt.plot(torch.stack(train_loss).detach().numpy())
# plt.show()
# plt.close()


# #%%
# torch.save(rgcn_params, f'saved_model/{model_name}_rgcn.pth')
# torch.save(link_perd_params, f'saved_model/{model_name}_link_pred.pth')


# #%%
# # rgcn.load_state_dict(rgcn_params)
# rgcn.load_state_dict(torch.load(f'saved_model/{model_name}_rgcn.pth'))
# rgcn.eval()

# # predictor.load_state_dict(link_perd_params)
# predictor.load_state_dict(torch.load(f'saved_model/{model_name}_link_pred.pth'))
# rgcn.eval()
