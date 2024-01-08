#%%
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

from ogb.linkproppred import LinkPropPredDataset, Evaluator

from tqdm.auto import tqdm
from collections import defaultdict


#%%
dataset_name = 'ogbl-biokg'
dataset = LinkPropPredDataset(name = dataset_name)
split_edge = dataset.get_edge_split()

train_triples, valid_triples, test_triples = split_edge['train'], split_edge['valid'], split_edge['test']

nrelation = int(max(train_triples['relation'])) + 1

entity_dict = dict()
cur_idx = 0
for key in dataset[0]['num_nodes_dict']:
    entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
    cur_idx += dataset[0]['num_nodes_dict'][key]
nentity = sum(dataset[0]['num_nodes_dict'].values())

evaluator = Evaluator(name = dataset_name)

train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--dataset', type=str, default='ogbl-biokg', help='dataset name, default to biokg')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000, help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500, help='number of negative samples when evaluating training triples')
    return parser.parse_args(args)

def log_metrics(mode, step, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        writer.add_scalar("_".join([mode, metric]), metrics[metric], step)


args = parse_args([])

warm_up_steps = max_step // 2

training_logs = []
for step in range(init_step, max_step):
    log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
    training_logs.append(log)
    
    if step >= warm_up_steps:
        current_learning_rate = current_learning_rate / 10
        logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        warm_up_steps = warm_up_steps * 3
            