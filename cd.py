#%%
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torch_geometric
from torch_geometric.nn import SAGEConv, HeteroConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


#%%
# raw 데이터 불러오기
chem_dis_path = 'dataset/raw/chem_dis.csv'
chem_dis_tmp = pd.read_csv(chem_dis_path, skiprows = list(range(27)) + [28])
chem_dis_tmp.DirectEvidence.value_counts()

target_dis_tmp = pd.read_excel('dataset/raw/CTD_disease_list.xlsx', sheet_name = ['target', 'hierarchy'])
target_dis_list = target_dis_tmp['target']
disease_hierarchy = target_dis_tmp['hierarchy']


# 각 데이터에서 사용할 column
chem_col = 'ChemicalID'
dis_col = 'DiseaseID'


#%%
''' all disease '''
# mapping of unique chemical, disease, and gene
uniq_chem = chem_dis_tmp[chem_col].unique()
chem_map = {name: i for i, name in enumerate(uniq_chem)}

uniq_dis = chem_dis_tmp[dis_col].unique()
dis_map = {name: i for i, name in enumerate(uniq_dis)}


# direct edge index between chem-disease
direct_chem_dis_idx = chem_dis_tmp.DirectEvidence == 'marker/mechanism'
direct_chem_dis = chem_dis_tmp[direct_chem_dis_idx][[chem_col, dis_col]]
assert direct_chem_dis.duplicated().sum() == 0

# inferenced relationship between chem-disease
curated_chem_dis_idx = chem_dis_tmp.DirectEvidence.isna()
curated_chem_dis = chem_dis_tmp[curated_chem_dis_idx][[chem_col, dis_col]]
curated_chem_dis = curated_chem_dis.drop_duplicates([chem_col, dis_col])

# drop the duplicated pair of chem-disease have direct evidence and inference score
dup_chem_idx = curated_chem_dis[chem_col].isin(direct_chem_dis[chem_col])
dup_dis_idx = curated_chem_dis[dis_col].isin(direct_chem_dis[dis_col])
curated_chem_dis = curated_chem_dis[~(dup_chem_idx & dup_dis_idx)]

# mapping the chemical and disease id
direct_chem_dis[chem_col] = direct_chem_dis[chem_col].apply(lambda x: chem_map[x])
direct_chem_dis[dis_col] = direct_chem_dis[dis_col].apply(lambda x: dis_map[x])

curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])


#%%
# create heterogeneous graph
data = HeteroData()

data['chemical'].id = torch.tensor(list(chem_map.values()))
data['chemical'].x = torch.tensor(list(chem_map.values()))
data['disease'].id = torch.tensor(list(dis_map.values()))
data['disease'].x = torch.tensor(list(dis_map.values()))

data['chemical', 'cause', 'disease'].edge_index = torch.from_numpy(direct_chem_dis.values.T).to(torch.long)
data['chemical', 'relate', 'disease'].edge_index = torch.from_numpy(curated_chem_dis.values.T).to(torch.long)
data['disease', 'rev_relate', 'chemical'].edge_index = torch.from_numpy(curated_chem_dis.values.T[[1, 0], :]).to(torch.long)


#%%
''' target disease: 7'''
target_dis_idx = chem_dis_tmp[dis_col].isin('MESH:'+target_dis_list['MESH ID'])
target_chem_dis = chem_dis_tmp[target_dis_idx]

# mapping of unique chemical, disease, and gene
uniq_chem = target_chem_dis[chem_col].unique()
chem_map = {name: i for i, name in enumerate(uniq_chem)}

uniq_dis = target_chem_dis[dis_col].unique()
dis_map = {name: i for i, name in enumerate(uniq_dis)}


# direct edge index between chem-disease
direct_chem_dis_idx = target_chem_dis.DirectEvidence == 'marker/mechanism'
direct_chem_dis = target_chem_dis[direct_chem_dis_idx][[chem_col, dis_col]]
assert direct_chem_dis.duplicated().sum() == 0

# inferenced relationship between chem-disease
curated_chem_dis_idx = target_chem_dis.DirectEvidence.isna()
curated_chem_dis = target_chem_dis[curated_chem_dis_idx][[chem_col, dis_col]]
curated_chem_dis = curated_chem_dis.drop_duplicates([chem_col, dis_col])

# drop the duplicated pair of chem-disease have direct evidence and inference score
dup_chem_idx = curated_chem_dis[chem_col].isin(direct_chem_dis[chem_col])
dup_dis_idx = curated_chem_dis[dis_col].isin(direct_chem_dis[dis_col])
curated_chem_dis = curated_chem_dis[~(dup_chem_idx & dup_dis_idx)]

# mapping the chemical and disease id
direct_chem_dis[chem_col] = direct_chem_dis[chem_col].apply(lambda x: chem_map[x])
direct_chem_dis[dis_col] = direct_chem_dis[dis_col].apply(lambda x: dis_map[x])

curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])



#%%
# create heterogeneous graph
data = HeteroData()

data['chemical'].id = torch.tensor(list(chem_map.values()))
data['chemical'].x = torch.tensor(list(chem_map.values()))
data['disease'].id = torch.tensor(list(dis_map.values()))
data['disease'].x = torch.tensor(list(dis_map.values()))

data['chemical', 'cause', 'disease'].edge_index = torch.from_numpy(direct_chem_dis.values.T).to(torch.long)
data['chemical', 'relate', 'disease'].edge_index = torch.from_numpy(curated_chem_dis.values.T).to(torch.long)
data['disease', 'rev_relate', 'chemical'].edge_index = torch.from_numpy(curated_chem_dis.values.T[[1, 0], :]).to(torch.long)


#%%
num_chem = len(data['chemical'].id)
num_disease = len(data['disease'].id)


class NodeEmbedding(nn.Module):
    def __init__(self, hidden_channels):
        super(NodeEmbedding, self).__init__()
        
        self.chem_emb = nn.Embedding(num_chem, hidden_channels)
        self.disease_emb = nn.Embedding(num_disease, hidden_channels)
        
    def forward(self, chem_id, dis_id):
        x = {
            'chemical': self.chem_emb(chem_id),
            'disease': self.disease_emb(dis_id)
        }
        return x
        

class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super(HeteroGNN, self).__init__()
        
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('chemical', 'cause', 'disease'): SAGEConv(hidden_channels, hidden_channels),
                ('chemical', 'relate', 'disease'): SAGEConv(hidden_channels, hidden_channels),
                ('disease', 'rev_relate', 'chemical'): SAGEConv(hidden_channels, hidden_channels)
            }, aggr = 'sum')
            self.convs.append(conv)
    
    def forward(self, x_dict, edge_index):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index)
            x_dict = {key: nn.functional.relu(x) for key, x in x_dict.items()}
            x_dict = {key: nn.functional.dropout(x, p = self.dropout, training = self.training) for key, x in x_dict.items()}
        x_dict = self.convs[-1](x_dict, edge_index)
        
        return x_dict


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
        
        node_embedding = node_embed(data['chemical'].id, data['disease'].id)
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
    num_test = 0.1,
    is_undirected = False,
    disjoint_train_ratio = 0.,
    # disjoint_train_ratio = 0.3,
    neg_sampling_ratio = 1.0,
    add_negative_train_samples = True,
    edge_types = ('chemical', 'cause', 'disease')
)

train_data, valid_data, test_data = data_split(data)

hidden_channels = 256
out_channels = 1
num_layer = 3
dropout = 0
batch_size = 512

node_embed = NodeEmbedding(hidden_channels)
gnn = HeteroGNN(hidden_channels, num_layer, dropout)
predictor = LinkPredictor(hidden_channels, out_channels, num_layer, dropout)


criterion = nn.BCELoss()
models = (node_embed, gnn, predictor)


param_groups = []
param_groups.append({'params': node_embed.parameters()})
param_groups.append({'params': gnn.parameters()})
param_groups.append({'params': predictor.parameters()})
optimizer = optim.Adam(param_groups, lr = 0.001)


epochs = 100
train_loss = []
for epoch in range(1, epochs+1):
    _train_loss = train(models, optimizer, criterion, train_data, batch_size)
    train_loss.append(_train_loss)
    
    # if epoch % 10 == 0:
    val_loss, val_auc = evaluation(models, criterion, valid_data)
    test_loss, test_auc = evaluation(models, criterion, test_data)
    
    print(f'=== epoch {epoch} ===')
    print(f'train loss: {_train_loss:.3f}, validation loss: {val_loss:.3f}, test loss: {test_loss:.3f}')
    print(f'validation auc: {val_auc:.3f}, test auc: {test_auc:.3f}')


#%%
plt.plot(train_loss)
plt.show()
plt.close()





#%%
''' target disease with hierarchy '''
disease_hierarchy = 'MESH:' + disease_hierarchy
target_dis_idx = chem_dis_tmp[dis_col].isin(np.unique(disease_hierarchy.values.reshape(-1)))
target_chem_dis = chem_dis_tmp[target_dis_idx]

# mapping of unique chemical, disease, and gene
uniq_chem = target_chem_dis[chem_col].unique()
chem_map = {name: i for i, name in enumerate(uniq_chem)}

uniq_dis = np.unique(np.concatenate([target_chem_dis[dis_col].values, disease_hierarchy.values.reshape(-1)]))
dis_map = {name: i for i, name in enumerate(uniq_dis)}


# direct edge index between chem-disease
direct_chem_dis_idx = target_chem_dis.DirectEvidence == 'marker/mechanism'
direct_chem_dis = target_chem_dis[direct_chem_dis_idx][[chem_col, dis_col]]
assert direct_chem_dis.duplicated().sum() == 0

# inferenced relationship between chem-disease
curated_chem_dis_idx = target_chem_dis.DirectEvidence.isna()
curated_chem_dis = target_chem_dis[curated_chem_dis_idx][[chem_col, dis_col]]
curated_chem_dis = curated_chem_dis.drop_duplicates([chem_col, dis_col])

# drop the duplicated pair of chem-disease have direct evidence and inference score
dup_chem_idx = curated_chem_dis[chem_col].isin(direct_chem_dis[chem_col])
dup_dis_idx = curated_chem_dis[dis_col].isin(direct_chem_dis[dis_col])
curated_chem_dis = curated_chem_dis[~(dup_chem_idx & dup_dis_idx)]

# mapping the chemical and disease id
direct_chem_dis[chem_col] = direct_chem_dis[chem_col].apply(lambda x: chem_map[x])
direct_chem_dis[dis_col] = direct_chem_dis[dis_col].apply(lambda x: dis_map[x])

curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])

disease_hierarchy['Parent ID'] = disease_hierarchy['Parent ID'].apply(lambda x: dis_map[x])
disease_hierarchy['Child ID'] = disease_hierarchy['Child ID'].apply(lambda x: dis_map[x])


#%%
# create heterogeneous graph
data = HeteroData()

data['chemical'].id = torch.tensor(list(chem_map.values()))
data['chemical'].x = torch.tensor(list(chem_map.values()))
data['disease'].id = torch.tensor(list(dis_map.values()))
data['disease'].x = torch.tensor(list(dis_map.values()))

data['chemical', 'cause', 'disease'].edge_index = torch.from_numpy(direct_chem_dis.values.T).to(torch.long)
data['chemical', 'relate', 'disease'].edge_index = torch.from_numpy(curated_chem_dis.values.T).to(torch.long)
data['disease', 'rev_relate', 'chemical'].edge_index = torch.from_numpy(curated_chem_dis.values.T[[1, 0], :]).to(torch.long)
data['disease', 'child', 'disease'].edge_index = torch.from_numpy(disease_hierarchy.values.T).to(torch.long)


#%%
num_chem = len(data['chemical'].id)
num_disease = len(data['disease'].id)


class NodeEmbedding(nn.Module):
    def __init__(self, hidden_channels):
        super(NodeEmbedding, self).__init__()
        
        self.chem_emb = nn.Embedding(num_chem, hidden_channels)
        self.disease_emb = nn.Embedding(num_disease, hidden_channels)
        
    def forward(self, chem_id, dis_id):
        x = {
            'chemical': self.chem_emb(chem_id),
            'disease': self.disease_emb(dis_id)
        }
        return x
        

class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super(HeteroGNN, self).__init__()
        
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('chemical', 'cause', 'disease'): SAGEConv(hidden_channels, hidden_channels),
                ('chemical', 'relate', 'disease'): SAGEConv(hidden_channels, hidden_channels),
                ('disease', 'rev_relate', 'chemical'): SAGEConv(hidden_channels, hidden_channels),
                ('disease', 'child', 'disease'): SAGEConv(hidden_channels, hidden_channels)
            }, aggr = 'sum')
            self.convs.append(conv)
    
    def forward(self, x_dict, edge_index):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index)
            x_dict = {key: nn.functional.relu(x) for key, x in x_dict.items()}
            x_dict = {key: nn.functional.dropout(x, p = self.dropout, training = self.training) for key, x in x_dict.items()}
        x_dict = self.convs[-1](x_dict, edge_index)
        
        return x_dict


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
        
        node_embedding = node_embed(data['chemical'].id, data['disease'].id)
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
    num_test = 0.1,
    is_undirected = False,
    disjoint_train_ratio = 0.3,
    neg_sampling_ratio = 1.0,
    add_negative_train_samples = True,
    edge_types = ('chemical', 'cause', 'disease')
)

train_data, valid_data, test_data = data_split(data)


hidden_channels = 256
out_channels = 1
num_layer = 3
dropout = 0
batch_size = 512

node_embed = NodeEmbedding(hidden_channels)
gnn = HeteroGNN(hidden_channels, num_layer, dropout)
predictor = LinkPredictor(hidden_channels, out_channels, num_layer, dropout)


criterion = nn.BCELoss()
models = (node_embed, gnn, predictor)


param_groups = []
param_groups.append({'params': node_embed.parameters()})
param_groups.append({'params': gnn.parameters()})
param_groups.append({'params': predictor.parameters()})
optimizer = optim.Adam(param_groups, lr = 0.001)


epochs = 100
train_loss = []
for epoch in range(1, epochs+1):
    _train_loss = train(models, optimizer, criterion, train_data, batch_size)
    train_loss.append(_train_loss)
    
    # if epoch % 10 == 0:
    val_loss, val_auc = evaluation(models, criterion, valid_data)
    test_loss, test_auc = evaluation(models, criterion, test_data)
    
    print(f'=== epoch {epoch} ===')
    print(f'train loss: {_train_loss:.3f}, validation loss: {val_loss:.3f}, test loss: {test_loss:.3f}')
    print(f'validation auc: {val_auc:.3f}, test auc: {test_auc:.3f}')


#%%
plt.plot(train_loss)
plt.show()
plt.close()

# %%
