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
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RandomLinkSplit

from torch_geometric.utils import k_hop_subgraph


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

uniq_dis = pd.concat([target_chem_dis[dis_col], target_gene_dis[dis_col]]).unique()
dis_map = {name: i for i, name in enumerate(uniq_dis)}

uniq_gene = pd.concat([target_gene_dis[gene_col], chem_gene_tmp[gene_col]]).unique()
gene_map = {str(name): i for i, name in enumerate(uniq_gene)}


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
    
    dataRGCN.edge_index = data_edge_index
    dataRGCN.x = datax
    
    edge_label_index =torch.tensor( heteroData["chemical","cause","disease"].edge_label_index.cpu().numpy())
    edge_label_index[1,:] += (len(heteroData["chemical"].x) + len(heteroData["gene"].x)  )
    edge_label = torch.tensor(heteroData["chemical","cause","disease"].edge_label.cpu().numpy())
    target_label_type = torch.zeros(len(edge_label)) + data.edge_types.index(("chemical","cause","disease"))
    dataRGCN.edge_label_index = edge_label_index
    dataRGCN.edge_label = edge_label
    dataRGCN.edge_type = data_edge_types
    dataRGCN.label_edge_type = target_label_type
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


a = RGCNEncoder(num_nodes, len(data.edge_types), num_layers = 2, emb_dim = 32)
embedding = a(rgcn_test.x.long(), rgcn_test.edge_index.long(), rgcn_test.edge_type.long())

src, dst = rgcn_test.edge_label_index

rgcn_test.label_edge_type


class DistMult(nn.Module):
    def __init__(self, num_relations, emb_dim):
        super(DistMult, self).__init__()
        
        self.relation_embedding = nn.Embedding(num_relations, emb_dim)

    def forward(self, x_i, x_j, edge_type):
        relation_embedding = self.relation_embedding(edge_type)
        score = torch.sum(x_i * relation_embedding * x_j, dim = -1)
        return score


distmult = DistMult(9, 32)
pred = distmult(embedding[src], embedding[dst], rgcn_test.label_edge_type.long())

pos_idx = rgcn_test.edge_label == 1
neg_idx = rgcn_test.edge_label == 0

pos_score = pred[pos_idx]
neg_score = pred[neg_idx]


nn.functional.margin_ranking_loss(pos_score, neg_score,target=torch.ones_like(pos_score))


class TransE(nn.Module):
    def __init__(self, num_relations, emb_dim):
        super(TransE, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, emb_dim)

    def forward(self, x_i, x_j, edge_type):
        relation_embedding = self.relation_embedding(edge_type)
        score = torch.norm(x_i + relation_embedding - x_j, p = 1, dim = -1)
        
        return score

transe = TransE(9, 32)
transe(embedding[src], embedding[dst], rgcn_test.label_edge_type.long())


class MultiLayerPerceptron(nn.Module):
    def __init__(self, emb_dim, out_channels, num_layers, dropout):
        super(MultiLayerPerceptron, self).__init__()
        
        self.lins = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(emb_dim, emb_dim))
        self.lins.append(nn.Linear(emb_dim, out_channels))
        
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
        node_embedding = gnn(node_embedding, data.edge_index_dict, data.edge_types)
        
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
def evaluation(models, criterion, data):
    node_embed, gnn, predictor = models
    
    node_embed.eval()
    gnn.eval()
    predictor.eval()
    
    src, dst = data.edge_label_index_dict[('chemical', 'cause', 'disease')]
    
    node_embedding = node_embed(data['chemical'].id, data['disease'].id)
    node_embedding = gnn(node_embedding, data.edge_index_dict)
    
    pred = predictor(node_embedding['chemical'][src], node_embedding['disease'][dst])
    target = data[('chemical', 'cause', 'disease')].edge_label
    
    loss = criterion(pred, target.view(pred.shape))
    auc = roc_auc_score(target.cpu().numpy(), pred.detach().cpu().numpy())
    
    return loss, auc





#%%
seed = 0

torch.manual_seed(seed)
torch_geometric.seed_everything(seed)

data_split = RandomLinkSplit(
    num_val = 0.1, 
    num_test = 0.01,
    is_undirected = False,
    disjoint_train_ratio = 0.,
    # disjoint_train_ratio = 0.3,
    neg_sampling_ratio = 1.0,
    add_negative_train_samples = True,
    edge_types = ('chemical', 'cause', 'disease')
)

train_data, valid_data, test_data = data_split(hetero_data)

rgcn_train = convert_hetero_object_to_RGCN(train_data, hetero_data.edge_types)
rgcn_valid = convert_hetero_object_to_RGCN(valid_data, hetero_data.edge_types)
rgcn_test = convert_hetero_object_to_RGCN(test_data, hetero_data.edge_types)


hidden_channels = 10
out_channels = 1
num_layer = 3
dropout = 0
batch_size = 512
num_relations = len(data.edge_index_dict)
num_blocks = 5
num_nodes = len(rgcn_train.x)

a = RGCNEncoder(num_nodes, len(data.edge_types), num_layers = 2, emb_dim = 32)
a(rgcn_test.x.long(), rgcn_test.edge_index.long(), rgcn_test.edge_type.long())


node_embed = NodeEmbedding(hidden_channels)
gnn = HeteroGNN(num_relations, num_blocks, hidden_channels, num_layer, dropout)
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