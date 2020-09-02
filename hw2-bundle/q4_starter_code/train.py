
import argparse
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import models
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')

    parser.set_defaults(model_type='GCN',
                        dataset='cora',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=32,
                        dropout=0.0,
                        epochs=200,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()

def train(dataset, task, args, writer, dev_str):
    if task == 'graph':
        dataset = dataset.shuffle()
        # graph classification: separate dataloader for test set
        data_size = len(dataset)
        loader = DataLoader(
                dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
                dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)
    elif task == 'node':
        # use mask to split train/validation/test
        test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise RuntimeError('Unknown task')

    dev = torch.device(dev_str)
    # build model
    model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task).to(dev)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    # train
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            batch = batch.to(dev)
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
#        print(total_loss)
        writer.add_scalar(dev_str+" loss "+task, total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model, dev)
#            print(test_acc,   '  test')
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar(dev_str+" test accuracy "+task, test_acc, epoch)

def test(loader, model, dev, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(dev)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total

def main():
    args = arg_parse()
    dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('CUDA availability:', torch.cuda.is_available())
#    dev = 'cpu'
    writer = SummaryWriter("./log/" + dev_str + '_' + args.dataset + '_' + args.model_type + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    if args.dataset == 'enzymes':
        dataset = TUDataset(root='../data/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='../data/Cora', name='Cora')
        ## 注意：split (string) – 默认是split='public'
        ## The type of dataset split ("public", "full", "random"). 
        ## If set to "public", the split will be the public fixed split from the “Revisiting Semi-Supervised Learning with Graph Embeddings” paper. 
        ## If set to "full", all nodes except those in the validation and test sets will be used for training (as in the “FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling” paper). 
        ## If set to "random", train, validation, and test sets will be randomly generated, according to num_train_per_class, num_val and num_test. (default: "public")
        task = 'node'
    train(dataset, task, args, writer, dev_str) 

if __name__ == '__main__':
    main()

