#%%
import argparse

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


#%%
''' Model '''
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize = False)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize = False)
            )
        
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize = False)
        )
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.convs[-1](x, adj_t)
        
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
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
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.lins[-1](x)
        
        return torch.sigmoid(x)


#%%
class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def reset(self, run):
        assert run >= 0 and run < len(self.results)
        self.results[run] = []

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.4f}')
            print(f'Highest Valid: {result[:, 1].max():.4f}')
            print(f'  Final Train: {result[argmax, 0]:.4f}')
            print(f'   Final Test: {result[argmax, 2]:.4f}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            print(best_result)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')


#%%
def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()
    
    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)
    
    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle = True):
        optimizer.zero_grad()
        
        h = model(data.x, data.adj_t)
        src, dst = source_edge[perm], target_edge[perm]
        
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        # just do some trivial random sampling
        dst_neg = torch.randint(0, data.num_nodes, src.size(), dtype = torch.long, device = h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        num_examples = pos_out.size(0)
        total_loss += (loss.item() * num_examples)
        total_examples += num_examples
    
    return total_loss / total_examples
        

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()
    
    h = model(data.x, data.adj_t)
    
    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)
        
        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim = 0)
        
        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr        


#%%
parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=64 * 1024)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args([])
print(args)


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygLinkPropPredDataset(name = 'ogbl-citation2', transform = T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

split_edge = dataset.get_edge_split()

torch.manual_seed(0)
idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]

split_edge['eval_train'] = {
    'source_node': split_edge['train']['source_node'][idx],
    'target_nodel': split_edge['train']['target_node'][idx],
    'target_node_neg': split_edge['valid']['target_node_neg'],
}

model = GCN(
    in_channels = data.num_features,
    hidden_channels = args.hidden_channels,
    out_channels = args.hidden_channels,
    num_layers = args.num_layers,
    dropout = args.dropout
)
model = model.to(device)

# pre compute GCN normalization
adj_t = data.adj_t.set_diag()
deg = adj_t.sum(dim = 1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
data.adj_t = adj_t

predictor = LinkPredictor(
    in_channels = args.hidden_channels,
    hidden_channels = args.hidden_channels,
    out_channels = 1,
    num_layers = args.num_layers,
    dropout = args.dropout
)
predictor = predictor.to(device)

evaluator = Evaluator(name = 'ogbl-citation2')
logger = Logger(1, args)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(predictor.parameters()),
    lr = args.lr
)


for run in range(args.runs):
    model.reset_parameters()
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        loss = train(model, predictor, data, split_edge, optimizer,
                        args.batch_size)
        print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if epoch % args.eval_steps == 0:
            result = test(model, predictor, data, split_edge, evaluator,
                            args.batch_size)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_mrr, valid_mrr, test_mrr = result
                print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {train_mrr:.4f}, '
                        f'Valid: {valid_mrr:.4f}, '
                        f'Test: {test_mrr:.4f}')

    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics(run)
print('GraphSAGE' if args.use_sage else 'GCN')
logger.print_statistics()


#%%
