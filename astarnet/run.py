#%%
''' Load Dataset '''
import os

import torch
from torch.utils import data as torch_data
from torch_geometric.data import Data


dataset = torch.load('../dataset/homo_ctd_graph.pt')
num_nodes = dataset.num_nodes
# num_relations = 1
num_relation = int(dataset.edge_type.max()) + 1
edge_index = dataset.edge_index[:, dataset.train_mask]
edge_type = dataset.edge_type[dataset.train_mask]

edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
edge_type = torch.cat([edge_type, edge_type + num_relation])
num_relation = num_relation * 2

train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                    target_edge_index=dataset.edge_index[:, dataset.train_mask],
                    target_edge_type=dataset.edge_type[dataset.train_mask])
valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                    target_edge_index=dataset.edge_index[:, dataset.val_mask],
                    target_edge_type=dataset.edge_type[dataset.val_mask])
test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                    target_edge_index=dataset.edge_index[:, dataset.test_mask],
                    target_edge_type=dataset.edge_type[dataset.test_mask])
# inmemory dataset module로 구성했을 겅유 collate function 사용 가능
# dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])

train_batch_size = 64
train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
train_loader = torch_data.DataLoader(train_triplets, train_batch_size, shuffle = True)

for _, batch in enumerate(train_loader):
    break


#%%
''' functional '''
import torch


def multikey_argsort(inputs, descending=False, break_tie=False):
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    return order


def bincount(input, minlength=0):
    if input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)

    sorted = (input.diff() >= 0).all()
    if sorted:
        if minlength == 0:
            minlength = input.max() + 1
        range = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range, input)
        return index.diff()

    return input.bincount(minlength=minlength)


def variadic_topks(input, size, ks, largest=True, break_tie=False):
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)

    range = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range = range + offset.repeat_interleave(ks)
    index = order[range]

    return input[index], index


#%%
''' data '''
import torch

from torch_scatter import scatter_add
from torch_sparse import spmm

from torchdrug import core, data, utils


allow_materialization = False


class VirtualTensor(object):

    def __init__(self, keys=None, values=None, index=None, input=None, shape=None, dtype=None, device=None):
        if shape is None:
            shape = index.shape + input.shape[1:]
        if index is None:
            index = torch.zeros(*shape[:1], dtype=torch.long, device=device)
        if input is None:
            input = torch.empty(1, *shape[1:], dtype=dtype, device=device)
        if keys is None:
            keys = torch.empty(0, dtype=torch.long, device=device)
        if values is None:
            values = torch.empty(0, *shape[1:], dtype=dtype, device=device)

        self.keys = keys
        self.values = values
        self.index = index
        self.input = input

    @classmethod
    def zeros(cls, *shape, dtype=None, device=None):
        input = torch.zeros(1, *shape[1:], dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def full(cls, shape, value, dtype=None, device=None):
        input = torch.full((1,) + shape[1:], value, dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def gather(cls, input, index):
        return cls(index=index, input=input, dtype=input.dtype, device=input.device)

    def clone(self):
        return VirtualTensor(self.keys.clone(), self.values.clone(), self.index.clone(), self.input.clone())

    @property
    def shape(self):
        return self.index.shape + self.input.shape[1:]

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)
        keys = indexes[0]

        assert keys.numel() == 0 or (keys.max() < len(self.index) and keys.min() >= 0)
        values = self.input[(self.index[keys],) + indexes[1:]]
        if len(self.keys) > 0:
            index = torch.bucketize(keys, self.keys)
            index = index.clamp(max=len(self.keys) - 1)
            indexes = (index,) + indexes[1:]
            found = keys == self.keys[index]
            indexes = tuple(index[found] for index in indexes)
            values[found] = self.values[indexes]
        return values

    def __setitem__(self, keys, values):
        new_keys, inverse = torch.cat([self.keys, keys]).unique(return_inverse=True)
        new_values = torch.zeros(len(new_keys), *self.shape[1:], dtype=self.dtype, device=self.device)
        new_values[inverse[:len(self.keys)]] = self.values
        new_values[inverse[len(self.keys):]] = values
        self.keys = new_keys
        self.values = new_values

    def __len__(self):
        return self.shape[0]


class View(object):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [x._contiguous if isinstance(x, View) else x for x in args]
        return func(*args, **kwargs)

    @utils.cached_property
    def _contiguous(self):
        return self.contiguous()

    def is_contiguous(self, *args, **kwargs):
        return False

    @property
    def ndim(self):
        return len(self.shape)

    def __getattr__(self, name):
        return getattr(self._contiguous, name)

    def __repr__(self):
        return repr(self._contiguous)

    def __len__(self):
        return self.shape[0]


class Range(View):

    def __init__(self, end, device=None):
        self.end = end
        self.shape = (end,)
        self.device = device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return torch.arange(end, device=self.device)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            assert len(index) == 1
            index = index[0]
        return torch.as_tensor(index, device=self.device)


class Repeat(View):

    def __init__(self, input, repeats):
        super(Repeat, self).__init__()
        self.input = input
        self.repeats = repeats
        self.shape = (int(repeats) * input.shape[0],) + input.shape[1:]
        self.device = input.device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return self.input.repeat([self.repeats] + [1] * (self.input.ndim - 1))

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if index[0].numel() > 0:
            assert index[0].max() < len(self)
        index = (index[0] % len(self.input),) + index[1:]
        return self.input[index]


class RepeatInterleave(View):

    def __init__(self, input, repeats):
        super(RepeatInterleave, self).__init__()
        self.input = input
        self.repeats = repeats
        self.shape = (input.shape[0] * int(repeats),) + input.shape[1:]
        self.device = input.device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return self.input.repeat_interleave(self.repeats, dim=0)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if index[0].numel() > 0:
            assert index[0].max() < len(self)
        index = (index[0] // self.repeats,) + index[1:]
        return self.input[index]


class Add(View):

    def __init__(self, input, other):
        super(Add, self).__init__()
        self.input = input
        self.other = other
        shape = []
        for d, (i, o) in enumerate(zip(input.shape, other.shape)):
            if i != o and min(i, o) > 1:
                raise RuntimeError("The size of tensor a (%d) must match the size of tensor b (%d) at non-singleton "
                                   "dimension %d" % (i, o, d))
            shape.append(max(i, o))
        self.shape = tuple(shape)
        self.device = input.device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return self.input.add(self.other)

    def __getitem__(self, index):
        return self.input[index] + self.other[index]


class RepeatGraph(data.PackedGraph):

    def __init__(self, graph, repeats, **kwargs):
        if not isinstance(graph, data.PackedGraph):
            graph = graph.pack([graph])
        core._MetaContainer.__init__(self, **kwargs)
        self.input = graph
        self.repeats = repeats

        # data.PackedGraph
        self.num_nodes = graph.num_nodes.repeat(repeats)
        self.num_edges = graph.num_edges.repeat(repeats)
        self.num_cum_nodes = self.num_nodes.cumsum(0)
        self.num_cum_edges = self.num_edges.cumsum(0)

        # data.Graph
        self.num_node = graph.num_node * repeats
        self.num_edge = graph.num_edge * repeats
        self.num_relation = graph.num_relation

    @property
    def _offsets(self):
        return RepeatInterleave(self.num_cum_nodes - self.num_nodes, self.input.num_edge)

    @property
    def edge_list(self):
        offsets = self.num_cum_nodes - self.num_nodes
        offsets = torch.stack([offsets, offsets, torch.zeros_like(offsets)], dim=-1)
        offsets = RepeatInterleave(offsets, self.input.num_edge)
        return Add(Repeat(self.input.edge_list, self.repeats), offsets)

    @utils.cached_property
    def adjacency(self):
        return utils.sparse_coo_tensor(self.edge_list.t(), self.edge_weight.contiguous(), self.shape)

    def edge_mask(self, index, compact=False):
        index = self._standarize_index(index, self.num_edge)
        num_edges = bincount(self.edge2graph[index], minlength=self.batch_size)
        edge_list = self.edge_list[index]
        if compact:
            node_index = edge_list[:, :2].flatten()
            node_index, inverse = node_index.unique(return_inverse=True)
            num_nodes = bincount(self.node2graph[node_index], minlength=self.batch_size)
            edge_list[:, :2] = inverse.view(-1, 2)
            data_dict, meta_dict = self.data_mask(node_index, index)
        else:
            num_nodes = self.num_nodes
            data_dict, meta_dict = self.data_mask(edge_index=index)

        return type(self.input)(edge_list, edge_weight=self.edge_weight[index], num_nodes=num_nodes,
                                num_edges=num_edges, num_relation=self.num_relation, offsets=self._offsets[index],
                                meta_dict=meta_dict, **data_dict)

    @utils.cached_property
    def neighbor_inverted_index(self):
        node_in = self.input.edge_list[:, 0]
        node_in, order = node_in.sort()
        degree_in = bincount(node_in, minlength=self.input.num_node)
        ends = degree_in.cumsum(0)
        starts = ends - degree_in
        ranges = torch.stack([starts, ends], dim=-1)
        offsets = RepeatInterleave(self.num_cum_edges - self.num_edges, self.input.num_edge)
        order = Add(Repeat(order, self.repeats), offsets)
        offsets = (self.num_cum_edges - self.num_edges).unsqueeze(-1).expand(-1, 2)
        offsets = RepeatInterleave(offsets, self.input.num_node)
        ranges = Add(Repeat(ranges, self.repeats), offsets)
        return order, ranges

    def neighbors(self, index):
        order, ranges = self.neighbor_inverted_index
        starts, ends = ranges[index].t()
        num_neighbors = ends - starts
        offsets = num_neighbors.cumsum(0) - num_neighbors
        ranges = torch.arange(num_neighbors.sum(), device=self.device)
        ranges = ranges + (starts - offsets).repeat_interleave(num_neighbors)
        edge_index = order[ranges]
        node_out = self.edge_list[edge_index, 1]
        return edge_index, node_out

    def num_neighbors(self, index):
        order, ranges = self.neighbor_inverted_index
        starts, ends = ranges[index].t()
        num_neighbors = ends - starts
        return num_neighbors

    def personalized_pagerank(self, index, alpha=0.8, num_iteration=20):
        node_in, node_out = self.input.edge_list.t()[:2]
        edge_weight = self.input.edge_weight
        edge_weight = edge_weight / (self.input.degree_in[node_in] + 1e-10)

        init = torch.zeros(self.num_node, device=self.device)
        init[index] = 1
        init = init.view(self.repeats, -1).t()
        ppr = init
        index = torch.stack([node_out, node_in])
        for i in range(num_iteration):
            ppr = spmm(index, edge_weight, self.input.num_node, self.input.num_node, ppr)
            ppr = ppr * alpha + init * (1 - alpha)
        return ppr.t().flatten()

    @utils.cached_property
    def node2graph(self):
        range = Range(self.batch_size, device=self.device)
        return RepeatInterleave(range, self.input.num_node)

    @utils.cached_property
    def edge2graph(self):
        range = Range(self.batch_size, device=self.device)
        return RepeatInterleave(range, self.input.num_edge)

    def __getattr__(self, name):
        if "input" in self.__dict__:
            attr = getattr(self.__dict__["input"], name)
            if isinstance(attr, torch.Tensor):
                return Repeat(attr, self.repeats)
            return attr
        raise AttributeError("`RepeatGraph` object has no attribute `%s`" % name)


#%%
''' Task '''
@R.register("task.KnowledgeGraphCompletion")
class KnowledgeGraphCompletionOGB(tasks.KnowledgeGraphCompletion, core.Configurable):

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        # self.evaluator = linkproppred.Evaluator(dataset.name)
        self.num_entity = dataset.num_nodes
        self.num_relation = 10
        fact_mask = torch.zeros(len(dataset), dtype=torch.bool)
        fact_mask[train_set.indices] = 1
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
        self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        return train_set, valid_set, test_set

    def predict(self, batch, all_loss=None, metric=None):
        batch_size = len(batch)

        if all_loss is None:
            # test
            h_index, t_index, r_index = batch.unbind(-1)
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        target = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # in case of GPU OOM
        return target.cpu()

    def evaluate(self, pred, target):
        is_positive = torch.zeros(pred.shape, dtype=torch.bool)
        is_positive.scatter_(-1, target.unsqueeze(-1), 1)
        pos_pred = pred[is_positive]
        neg_pred = pred[~is_positive].view(len(pos_pred), -1)
        metric = self.evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})

        new_metric = {}
        for key in metric:
            new_key = key.split("_")[0]
            new_metric[new_key] = metric[key].mean()

        return new_metric

#%%
''' Layer '''
import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("layer.NBFNetConv")
class GeneralizedRelationalConv(layers.MessagePassingBase, core.Configurable):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        else:
            self.relation = nn.Embedding(num_relation * 2, input_dim)

        # force rspmm to be compiled, to avoid cold start in time benchmark
        adjacency = torch.rand(1, 1, 1, device=self.device).to_sparse()
        relation_input = torch.rand(1, 32, device=self.device)
        input = torch.rand(1, 32, device=self.device)
        functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul="mul")

    def compute_message(self, node_input, edge_input):
        if self.message_func == "transe":
            return node_input + edge_input
        elif self.message_func == "distmult":
            return node_input * edge_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            return torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation * 2

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
        else:
            relation_input = relation_input.transpose(0, 1)

        node_input = input[node_in]
        edge_input = relation_input[relation]
        message = self.compute_message(node_input, edge_input)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        batch_size = len(graph.query)
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = getattr(graph, "pna_degree_out", graph.degree_out)
        degree_out = degree_out.unsqueeze(-1) + 1
        if not isinstance(graph, data.PackedGraph):
            edge_weight = edge_weight.unsqueeze(-1)
            degree_out = degree_out.unsqueeze(-1)

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            degree_mean = getattr(graph, "pna_degree_mean", scale.mean())
            scale = scale / degree_mean
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        if not isinstance(graph, data.PackedGraph):
            update = update.view(len(update), batch_size, -1)
        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation * 2

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)
        node_in, node_out, relation = graph.edge_list.t()

        degree_out = getattr(graph, "pna_degree_out", graph.degree_out)
        degree_out = degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out, relation]), graph.edge_weight,
                                                (graph.num_node, graph.num_node, batch_size * graph.num_relation))
        else:
            relation_input = relation_input.transpose(0, 1).flatten(1)
            adjacency = graph.adjacency
        adjacency = adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            degree_mean = getattr(graph, "pna_degree_mean", scale.mean())
            scale = scale / degree_mean
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        if not isinstance(graph, data.PackedGraph):
            update = update.view(len(update), batch_size, -1)
        return update

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


@R.register("layer.CompGCNConv")
class CompositionalGraphConv(layers.MessagePassingBase, core.Configurable):

    message2mul = {
        "sub": "add",
        "mult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, message_func="mult", layer_norm=False, activation="relu"):
        super(CompositionalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.message_func = message_func

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.loop_relation = nn.Embedding(1, input_dim)
        self.linear = nn.Linear(3 * input_dim, output_dim)
        self.relation = nn.Embedding(num_relation * 2, input_dim)
        self.relation_linear = nn.Linear(input_dim, output_dim)

    def compute_message(self, node_input, edge_input):
        if self.message_func == "sub":
            return node_input - edge_input
        elif self.message_func == "mult":
            return node_input * edge_input
        elif self.message_func == "corr":
            node_input = torch.fft.rfft(node_input)
            edge_input = torch.fft.rfft(edge_input)
            return torch.fft.irfft(node_input.conj() * edge_input, n=input.shape[-1])
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation * 2

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        loop = torch.zeros(graph.num_node, dtype=torch.long, device=self.device)
        relation_input = self.relation.weight.expand(len(graph), -1, -1)
        relation_input = getattr(graph, "relation_input", relation_input)
        loop_relation = self.loop_relation.weight
        with graph.graph():
            graph.relation_input = self.relation_linear(relation_input)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
        else:
            relation_input = relation_input.expand(batch_size, -1, -1).transpose(0, 1)
            loop_relation = loop_relation.expand(1, batch_size, -1)

        shape = [graph.num_node] + [-1] * (loop_relation.ndim - 1)
        node_input = torch.cat([input[node_in], input])
        edge_input = torch.cat([relation_input[relation], loop_relation.expand(shape)])
        message = self.compute_message(node_input, edge_input)

        return message

    def aggregate(self, graph, message):
        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        edge_weight = graph.edge_weight * 2 / (graph.degree_in[node_in] * graph.degree_out[node_out]) ** 0.5
        edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        direction = torch.div(relation, self.num_relation, rounding_mode="floor")
        node_out = node_out * 3 + direction
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=self.device) * 3 + 2])
        if not isinstance(graph, data.PackedGraph):
            edge_weight = edge_weight.unsqueeze(-1)

        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * 3)
        update = update.unflatten(0, (graph.num_node, 3))
        if not isinstance(graph, data.PackedGraph):
            update = update.transpose(1, 2)
        update = update.flatten(-2)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "corr":
            return super(CompositionalGraphConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation * 2

        batch_size = len(graph.query)
        input = input.flatten(1)
        node_in, node_out, relation = graph.edge_list.t()

        relation_input = self.relation.weight.expand(len(graph), -1, -1)
        relation_input = getattr(graph, "relation_input", relation_input)
        loop_relation = self.loop_relation.weight
        with graph.graph():
            graph.relation_input = self.relation_linear(relation_input)

        edge_weight = graph.edge_weight * 2 / (graph.degree_in[node_in] * graph.degree_out[node_out]) ** 0.5
        edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=self.device)])
        direction = torch.div(relation, self.num_relation, rounding_mode="floor")
        node_out = torch.cat([node_out * 3 + direction, torch.arange(graph.num_node, device=self.device) * 3 + 2])
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            loop = torch.ones(graph.num_node, dtype=torch.long, device=self.device) * batch_size * graph.num_relation
            relation = torch.cat([relation, loop])
            relation_input = torch.cat([relation_input.flatten(0, 1), loop_relation])
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out, relation]), edge_weight,
                                                (graph.num_node, graph.num_node * 3, batch_size * graph.num_relation + 1))
        else:
            loop = torch.ones(graph.num_node, dtype=torch.long, device=self.device) * graph.num_relation
            relation = torch.cat([relation, loop])
            relation_input = self.relation.weight.expand(batch_size, -1, -1).transpose(0, 1).flatten(1)
            loop_relation = loop_relation.expand(1, batch_size, -1).flatten(1)
            relation_input = torch.cat([relation_input, loop_relation])
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out, relation]), edge_weight,
                                                (graph.num_node, graph.num_node * 3, graph.num_relation + 1))
        adjacency = adjacency.transpose(0, 1)

        if self.message_func == "sub":
            relation_input = -relation_input
        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
        update = update.unflatten(0, (graph.num_node, 3))
        if not isinstance(graph, data.PackedGraph):
            update = update.unflatten(-1, (batch_size, -1)).transpose(1, 2)
        update = update.flatten(-2)

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


#%%
import math

import torch
from torch import nn, autograd
from torch.nn import functional as F

from torch_scatter import segment_add_coo, scatter_add, scatter_max

from torchdrug import core, data, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, base_layer, num_layer, short_cut=False, concat_hidden=False, num_mlp_layer=2,
                 remove_one_hop=False, shared_graph=True, edge_dropout=0, num_beam=10, path_topk=10):
        super(NeuralBellmanFordNetwork, self).__init__()

        self.num_relation = base_layer.num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.shared_graph = shared_graph
        self.num_beam = num_beam
        self.path_topk = path_topk

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = None

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(core.Configurable.load_config_dict(base_layer.config_dict()))
        feature_dim = base_layer.output_dim * (num_layer if concat_hidden else 1) + base_layer.input_dim
        self.query = nn.Embedding(base_layer.num_relation * 2, base_layer.input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def search(self, graph, h_index, r_index, edge_grad=False, all_loss=None, metric=None):
        query = self.query(r_index)
        boundary = self.indicator(graph, h_index, query)
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        graphs = []
        layer_input = boundary
        for layer in self.layers:
            if edge_grad:
                graph = graph.clone().detach().requires_grad_()
            hidden = layer(graph, layer_input)
            if self.short_cut:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            graphs.append(graph)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        if isinstance(graph, data.PackedGraph):
            node_query = query.repeat_interleave(graph.num_nodes, dim=0)
        else:
            node_query = query.expand(graph.num_node, -1, -1)
        score = self.score(hidden, node_query)

        return {
            "node_feature": hidden,
            "node_score": score,
            "step_graphs": graphs,
        }

    def indicator(self, graph, index, query):
        if isinstance(graph, data.PackedGraph):
            boundary = torch.zeros(graph.num_node, *query.shape[1:], device=self.device)
            boundary[index] = query
        else:
            boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
            index = index.unsqueeze(-1).expand_as(query)
            boundary.scatter_(0, index.unsqueeze(0), query.unsqueeze(0))
        return boundary

    def score(self, hidden, node_query):
        hidden = torch.cat([hidden, node_query], dim=-1)
        score = self.mlp(hidden).squeeze(-1)
        return score

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        if self.edge_dropout:
            graph = graph.clone()
            graph._edge_weight = self.edge_dropout(graph.edge_weight)
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        if not self.shared_graph:
            batch_size = len(h_index)
            graph = RepeatGraph(graph, batch_size)
            offset = graph.num_cum_nodes - graph.num_nodes
            h_index = h_index + offset.unsqueeze(-1)
            t_index = t_index + offset.unsqueeze(-1)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.search(graph, h_index[:, 0], r_index[:, 0], all_loss=all_loss, metric=metric)
        score = output["node_score"]
        if self.shared_graph:
            score = score.transpose(0, 1).gather(1, t_index)
        else:
            score = score[t_index]

        return score

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)
        batch_size = len(h_index)
        graph = graph.repeat(batch_size)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset
        t_index = t_index + offset

        output = self.search(graph, h_index, r_index, edge_grad=True)
        score = output["node_score"]
        graphs = output["step_graphs"]
        score = score[t_index]
        edge_weights = [graph.edge_weight for graph in graphs]
        edge_grads = autograd.grad(score.sum(), edge_weights)
        for graph, edge_grad in zip(graphs, edge_grads):
            with graph.edge():
                graph.edge_grad = edge_grad
        lengths, source_indexes = self.beam_search_length(graphs, h_index, t_index)
        paths, weights, num_steps = self.topk_average_length(graph, lengths, source_indexes, t_index)

        return paths, weights, num_steps

    def beam_search_length(self, graphs, h_index, t_index):
        inf = float("inf")
        input = torch.full((graphs[0].num_node, self.num_beam), -inf, device=self.device)
        input[h_index, 0] = 0

        lengths = []
        source_indexes = []
        for graph in graphs:
            edge_mask = graph.edge_list[:, 0] != t_index[graph.edge2graph]
            node_in, node_out = graph.edge_list[edge_mask, :2].t()

            message = input[node_in] + graph.edge_grad[edge_mask].unsqueeze(-1)
            edge_index = torch.arange(graph.num_edge, device=self.device)[edge_mask]
            beam_index = torch.arange(self.num_beam, device=self.device)
            edge_index, beam_index = torch.meshgrid(edge_index, beam_index)
            source_index = torch.stack([edge_index, beam_index], dim=-1)

            node_out, order = node_out.sort()
            num_messages = bincount(node_out, minlength=graph.num_node) * self.num_beam
            message = message[order].flatten()
            source_index = source_index[order].flatten(0, -2)
            ks = num_messages.clamp(max=self.num_beam)
            length, index = variadic_topks(message, num_messages, ks)
            source_index = source_index[index]
            length = functional.variadic_to_padded(length, ks, value=-inf)[0]
            source_index = functional.variadic_to_padded(source_index, ks)[0]

            lengths.append(length)
            source_indexes.append(source_index)
            input = length

        return lengths, source_indexes

    def topk_average_length(self, graph, lengths, source_indexes, t_index):
        num_layer = len(self.layers)
        weights = []
        num_steps = []
        beam_indexes = []
        for i, length in enumerate(lengths):
            weight = length[t_index] / (i + 1)
            num_step = torch.full(weight.shape, i + 1, device=self.device)
            beam_index = torch.arange(self.num_beam, device=self.device).expand_as(weight)
            weights.append(weight)
            num_steps.append(num_step)
            beam_indexes.append(beam_index)
        weights = torch.cat(weights, dim=-1)
        num_steps = torch.cat(num_steps, dim=-1)
        beam_index = torch.cat(beam_indexes, dim=-1)
        weights, index = weights.topk(self.path_topk)
        num_steps = num_steps.gather(-1, index)
        beam_index = beam_index.gather(-1, index)

        paths = []
        t_index = t_index.unsqueeze(-1).expand_as(beam_index)
        for i in range(num_layer)[::-1]:
            mask = num_steps > i
            edge_index, new_beam_index = source_indexes[i][t_index, beam_index].unbind(dim=-1)
            edge_index = torch.where(mask, edge_index, 0)
            edges = graph.edge_list[edge_index]
            edges[:, :, :2] -= graph._offsets[edge_index].unsqueeze(-1)
            paths.append(edges)
            t_index = torch.where(mask, graph.edge_list[edge_index, 0], t_index)
            beam_index = torch.where(mask, new_beam_index, beam_index)
        paths = torch.stack(paths[::-1], dim=-2)

        return paths, weights, num_steps

    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            any = -torch.ones_like(h_index_ext)
            pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
        else:
            pattern = torch.stack([h_index, t_index, r_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index


@R.register("model.AStarNet")
class AStarNetwork(NeuralBellmanFordNetwork, core.Configurable):

    def __init__(self, base_layer, num_layer, indicator_func="onehot", short_cut=False, num_mlp_layer=2,
                 num_indicator_bin=10, node_ratio=0.1, degree_ratio=1, test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, **kwargs):
        for k in ["concat_hidden", "shared_graph"]:
            if k in kwargs:
                raise TypeError("`%s` is not supported by AStarNet" % k)
        super(AStarNetwork, self).__init__(base_layer, num_layer, short_cut, num_mlp_layer=num_mlp_layer,
                                           shared_graph=False, **kwargs)

        assert not self.concat_hidden
        self.indicator_func = indicator_func
        self.num_indicator_bin = num_indicator_bin
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie

        if indicator_func == "ppr":
            self.distance = nn.Embedding(num_indicator_bin, base_layer.input_dim)
        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        self.mlp = layers.MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        ks = (node_ratio * graph.num_nodes).long()
        es = (degree_ratio * ks * graph.num_edges / graph.num_nodes).long()

        node_in = score.keys
        num_nodes = bincount(graph.node2graph[node_in], minlength=len(graph))
        ks = torch.min(ks, num_nodes)
        score_in = score[node_in]
        index = variadic_topks(score_in, num_nodes, ks=ks, break_tie=self.break_tie)[1]
        node_in = node_in[index]
        num_nodes = ks

        num_neighbors = graph.num_neighbors(node_in)
        num_edges = scatter_add(num_neighbors, graph.node2graph[node_in], dim_size=len(graph))
        es = torch.min(es, num_edges)
        # chunk batch to reduce peak memory usage
        num_edge_mean = num_edges.float().mean().clamp(min=1)
        chunk_size = max(int(1e7 / num_edge_mean), 1)
        num_nodes = num_nodes.split(chunk_size)
        num_edges = num_edges.split(chunk_size)
        es = es.split(chunk_size)
        num_chunk_nodes = [num_node.sum() for num_node in num_nodes]
        node_ins = node_in.split(num_chunk_nodes)

        edge_indexes = []
        for node_in, num_node, num_edge, e in zip(node_ins, num_nodes, num_edges, es):
            edge_index, node_out = graph.neighbors(node_in)
            score_edge = score[node_out]
            index = variadic_topks(score_edge, num_edge, ks=e, break_tie=self.break_tie)[1]
            edge_index = edge_index[index]
            edge_indexes.append(edge_index)
        edge_index = torch.cat(edge_indexes)

        return edge_index

    def search(self, graph, h_index, r_index, all_loss=None, metric=None):
        query = self.query(r_index)
        boundary, score = self.indicator(graph, h_index, query)
        hidden = boundary.clone()
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary
            graph.hidden = hidden
            graph.score = score
            graph.node_id = Range(graph.num_node, device=self.device)
            graph.pna_degree_out = graph.degree_out
        with graph.edge():
            graph.edge_id = Range(graph.num_edge, device=self.device)
        pna_degree_mean = (graph[0].degree_out + 1).log().mean()

        num_nodes = []
        num_edges = []
        subgraphs = []
        for layer in self.layers:
            edge_index = self.select_edges(graph, graph.score)
            subgraph = graph.edge_mask(edge_index, compact=True)
            subgraph.pna_degree_mean = pna_degree_mean

            layer_input = F.sigmoid(subgraph.score).unsqueeze(-1) * subgraph.hidden
            hidden = layer(subgraph, layer_input)
            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]
            if self.short_cut:
                graph.hidden[node_out] = graph.hidden[node_out] + hidden[out_mask]
            else:
                graph.hidden[node_out] = hidden[out_mask]
            index = graph.node2graph[node_out]
            graph.score[node_out] = self.score(graph.hidden[node_out], query[index])

            # update graph-level attributes
            data_dict, meta_dict = subgraph.data_by_meta("graph")
            graph.meta_dict.update(meta_dict)
            graph.__dict__.update(data_dict)

            num_nodes.append(subgraph.num_nodes.float().mean())
            num_edges.append(subgraph.num_edges.float().mean())
            subgraphs.append(subgraph)

        if metric is not None:
            metric["#node per layer"] = torch.stack(num_nodes).mean()
            metric["#edge per layer"] = torch.stack(num_edges).mean()

        return {
            "node_feature": graph.hidden,
            "node_score": graph.score,
            "step_graphs": subgraphs,
        }

    def indicator(self, graph, index, query):
        if self.indicator_func == "onehot":
            boundary = VirtualTensor.zeros(graph.num_node, query.shape[1], device=self.device)
            boundary[index] = query
            score = VirtualTensor.gather(self.score(torch.zeros_like(query), query), graph.node2graph)
            score[index] = self.score(query, query)
        elif self.indicator_func == "ppr":
            ppr = graph.personalized_pagerank(index)
            bin = torch.logspace(-1, 0, self.num_indicator_bin, base=graph.num_node, device=self.device)
            bin_index = torch.bucketize(ppr, bin)
            distance = self.distance.weight
            boundary = VirtualTensor.gather(distance, bin_index)
            boundary[index] = query
            hidden = distance.repeat(len(graph), 1)
            node_query = query.repeat_interleave(self.num_indicator_bin, dim=0)
            score_index = bin_index + torch.repeat_interleave(graph.num_nodes) * self.num_indicator_bin
            score = VirtualTensor.gather(self.score(hidden, node_query), score_index)
            score[index] = self.score(query, query)
        else:
            raise ValueError("Unknown indicator function `%s`" % self.indicator_func)
        return boundary, score

    def score(self, hidden, node_query):
        heuristic = self.linear(torch.cat([hidden, node_query], dim=-1))
        x = self.layers[0].compute_message(hidden, heuristic)
        score = self.mlp(x).squeeze(-1)
        return score

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        if self.training:
            return super(AStarNetwork, self).forward(graph, h_index, t_index, r_index, all_loss, metric)

        # adjust batch size for test node ratio
        num_chunk = math.ceil(self.test_node_ratio / self.node_ratio / 5)
        h_indexes = h_index.chunk(num_chunk)
        t_indexes = t_index.chunk(num_chunk)
        r_indexes = r_index.chunk(num_chunk)
        scores = []
        for h_index, t_index, r_index in zip(h_indexes, t_indexes, r_indexes):
            score = super(AStarNetwork, self).forward(graph, h_index, t_index, r_index, all_loss, metric)
            scores.append(score)
        score = torch.cat(scores)
        return score

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)
        batch_size = len(h_index)
        graph = RepeatGraph(graph, batch_size)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset
        t_index = t_index + offset

        output = self.search(graph, h_index, r_index)
        subgraphs = output["step_graphs"]
        lengths, source_indexes = self.beam_search_length(graph, subgraphs, graph.num_node, h_index, t_index)
        paths, weights, num_steps = self.topk_average_length(graph, lengths, source_indexes, t_index)

        return paths, weights, num_steps

    def beam_search_length(self, graph, subgraphs, num_node, h_index, t_index):
        inf = float("inf")
        input = VirtualTensor.full((num_node, self.num_beam), -inf, device=self.device)
        init = torch.full((len(h_index), self.num_beam), -inf, device=self.device)
        init[:, 0] = 0
        input[h_index] = init

        lengths = []
        source_indexes = []
        for subgraph in subgraphs:
            edge_mask = subgraph.node_id[subgraph.edge_list[:, 0]] != t_index[subgraph.edge2graph]
            node_in, node_out = subgraph.edge_list[edge_mask, :2].t()

            in_mask = subgraph.degree_in > 0
            sub_input = input[subgraph.node_id]
            score = F.sigmoid(subgraph.score) * in_mask
            score = score / scatter_max(score, subgraph.node2graph)[0][subgraph.node2graph]
            message = sub_input[node_in] + score[node_in].unsqueeze(-1)
            edge_index = subgraph.edge_id[edge_mask]
            beam_index = torch.arange(self.num_beam, device=self.device)
            edge_index, beam_index = torch.meshgrid(edge_index, beam_index)
            sub_source_index = torch.stack([edge_index, beam_index], dim=-1)

            node_out, order = node_out.sort()
            num_messages = bincount(node_out, minlength=subgraph.num_node) * self.num_beam
            message = message[order].flatten()
            sub_source_index = sub_source_index[order].flatten(0, -2)
            ks = num_messages.clamp(max=self.num_beam)
            sub_length, index = variadic_topks(message, num_messages, ks)
            sub_source_index = sub_source_index[index]
            sub_length = functional.variadic_to_padded(sub_length, ks, value=-inf)[0]
            sub_source_index = functional.variadic_to_padded(sub_source_index, ks)[0]

            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]
            length = VirtualTensor.full((num_node, self.num_beam), -inf, device=self.device)
            source_index = VirtualTensor.zeros(num_node, self.num_beam, 2, dtype=torch.long, device=self.device)
            length[node_out] = sub_length[out_mask]
            source_index[node_out] = sub_source_index[out_mask]

            lengths.append(length)
            source_indexes.append(source_index)
            input = length

        return lengths, source_indexes


#%%
''' task - negative sampling '''
from functools import reduce

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)
# batch = negative_sampling(train_data, batch, num_negative = train_batch_size, strict = True)


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


#%%
import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm

logger = logging.getLogger(__file__)


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    # template = jinja2.Template(raw)
    # instance = template.render(context)
    cfg = yaml.safe_load(raw)
    cfg = easydict.EasyDict(cfg)
    return cfg


cfg = load_config('ctd_config.yaml', context=vars)

def build_solver(cfg, dataset):
    # train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_data), len(valid_data), len(test_data)))

    task = core.Configurable.load_config_dict(cfg.task)
    # cfg.optimizer.params = task.parameters()
    # optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    optimizer = torch.optim.Adam()
    model = AStarNetwork(base_layer = GeneralizedRelationalConv(input_dim = 32, output_dim = 32, num_relation = num_relation, query_input_dim = 32), num_layer = 3)

    if "checkpoint" in cfg:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.checkpoint)
        checkpoint = os.path.expanduser(cfg.checkpoint)
        state = torch.load(cfg.checkpoint, map_location=solver.device)
        state["model"] = {k: v for k, v in state["model"].items() if isinstance(v, torch.Tensor)}

        solver.model.load_state_dict(state["model"], strict=False)
        solver.optimizer.load_state_dict(state["optimizer"])
        for state in solver.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(solver.device)

        comm.synchronize()

    return solver

#%%
import os
import sys
import math

import torch

from torchdrug import core, tasks
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

num_epoch = 20
def train_and_validate(cfg, solver):
    if num_epoch == 0:
        return

    if hasattr(cfg.train, "batch_per_epoch"):
        step = 1
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pretty.format(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    train_and_validate(cfg, solver)
    test(cfg, solver)