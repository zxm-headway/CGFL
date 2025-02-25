from __future__ import division, print_function

import argparse
import json
import os
import random
import time

# import tensorflow as tf
import torch
import torch.nn as nn
from sklearn import metrics
import deepdish as dd
from tqdm import trange

from models_pytorch import TGCN
# from models_bert import TGCN2
# from torch_geometric.data.sampler import NeighborSampler
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--no_sparse', action='store_true')
    parser.add_argument("--load_ckpt", action='store_true')
    parser.add_argument('--featureless', action='store_true')
    parser.add_argument("--save_path", type=str, default='./saved_model', help="the path of saved model")
    parser.add_argument('--dataset', type=str, default='post', help='dataset name, default to post')
    parser.add_argument('--model', type=str, default='gcn', help='model name, default to gcn')
    parser.add_argument('--lr', '--learning_rate', default=0.00002, type=float)   # 0.002/0.0002
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument("--hidden", default=512, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)   # 0.5/0.3/0.1
    parser.add_argument("--weight_decay", default=0.000001, type=float)
    parser.add_argument("--early_stop", default=2000, type=int)
    parser.add_argument("--num_graph", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(args)


def save_model(model, optimizer, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.save_path, 'model.bin'))


def train(args, features, train_label, train_mask, val_label, val_mask, test_label, test_mask, model, indice_list, weight_list):
    cost_valid = []
    acc_valid = []
    max_acc = 0.0
    min_cost = 10.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    model.train()
    best_FS = 0
    for epoch in range(args.epochs):
        # print('Epoch:', epoch)
        t = time.time()
        # print('Epoch:', epoch)
        outs = model(features, indice_list, weight_list, 1-args.dropout)
        pre_loss = loss_fct(outs, train_label)
        # pre_loss = loss_function(outs, train_label,train_mask, expt_type=5, scale=2)
        train_pred = torch.argmax(outs, dim=-1)

        ce_loss = (pre_loss * train_mask/train_mask.mean()).mean()
        train_acc = ((train_pred == train_label).float() * train_mask/train_mask.mean()).mean()
        # loss = ce_loss + tmp_loss
        loss = ce_loss
        # loss = pre_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        model.eval()
        # Validation
        valid_cost, valid_acc, pred, labels, duration,M = evaluate(args,features, val_label, val_mask, model, indice_list, weight_list)
        
        # if valid_acc >= best_FS and epoch > 200:
        if valid_acc > best_FS and epoch > 300:
            best_FS = valid_acc
            save_model(model, optimizer, args)
            min_cost = cost_valid[-1]
            print("Current best loss {:.5f}".format(min_cost))

        # Testing
        test_cost, test_acc, pred, labels, test_duration,_ = evaluate(args,features, test_label, test_mask, model, indice_list, weight_list)
        model.train()
        cost_valid.append(valid_cost)
        acc_valid.append(valid_acc)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
            "{:.5f}".format(train_acc.item()), "val_loss=", "{:.5f}".format(valid_cost),
            "val_acc=", "{:.5f}".format(valid_acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=",
            "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))

        # save model
        # if epoch > 700 and cost_valid[-1] < min_cost:
        # if epoch > 300 and M[2] > best_FS:
        # best_FS = M[2]
  

        # if acc_valid[-1] > max_acc:
        #     save_model(model, optimizer, args)
        #     min_cost = cost_valid[-1]
        #     max_acc = acc_valid[-1]
        #     print("Current best acc {:.5f}".format(max_acc))

        if epoch > args.early_stop and cost_valid[-1] > np.mean(cost_valid[-(args.early_stop + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

def evaluate(args, features, label, mask, model, indice_list, weight_list):
    t_test = time.time()
    # loss_fct = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        outs = model(features, indice_list, weight_list, 1)
        # pre_loss = loss_fct(outs, label)
        pre_loss = loss_function(outs, label,mask, expt_type=5, scale=2)
        pred = torch.argmax(outs, dim=-1)
        ce_loss = (pre_loss * mask/mask.mean()).mean()
        loss = ce_loss
        # loss = pre_loss
        acc = ((pred == label).float() * mask/mask.mean()).mean()

        M = gr_metrics(pred, label, mask)


    # feed_dict_val = construct_feed_dict(
    #     features, support, support_mix, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return loss.item(), acc.item(), pred.cpu().numpy(), label.cpu().numpy(), (time.time() - t_test),M

def load_ckpt(model):
    model_dict = model.state_dict()
    pretrained_dict = dd.io.load('./gcn.h5')

    print(pretrained_dict.keys())
    model_dict['layers.0.intra_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_0:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_00:0'].T, dtype=torch.float)
    model_dict['layers.0.intra_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_1:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_11:0'].T, dtype=torch.float)
    model_dict['layers.0.intra_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_2:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_22:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_0:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_00:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_1:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_11:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_2:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_22:0'].T, dtype=torch.float)    
    
    model.load_state_dict(model_dict)

# tf.compat.v1.disable_eager_execution()
def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data


def set_seed(args):
    """
    :param args:
    :return:
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def main(args):
    
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    f_file = os.sep.join(['..', 'data_tgcn', args.dataset, 'build_train'])
    if torch.cuda.is_available():
        device = 'cuda'
    set_seed(args)


    # Load data
    adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels,vocab = load_corpus_torch(args.dataset, device)
    # adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels,vocab = load_corpus_torch(args.dataset, device)
    # adj, adj1, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels = load_corpus_torch(args.dataset, device)
    adj = adj.tocoo()
    adj1 = adj1.tocoo()
    adj2 = adj2.tocoo()

    # support_mix = [adj, adj1, adj2]
    # support_mix = [adj]
    support_mix = [adj, adj1, adj2]

    indice_list, weight_list = [] , []
    for adjacency in support_mix:
        ind, dat = get_edge_tensor(adjacency)
        indice_list.append(ind.to(device))
        weight_list.append(dat.to(device))
        
    in_dim = adj.shape[0]
    # print(in_dim,'node_size')

    # print('Feature dim:', ou)
    # model = TGCN(in_dim=in_dim,hidden_dim=args.hidden, out_dim=num_labels,
    #     num_graphs=args.num_graph, dropout=args.dropout, n_layers=args.layers, bias=False, featureless=args.featureless)
    
    print('Starting model initialization...')

    # model = TGCN(in_dim=in_dim, vob=vocab, hidden_dim=args.hidden, out_dim=num_labels,
    model = TGCN(in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_labels,
                    num_graphs=args.num_graph, dropout=args.dropout, n_layers=args.layers, bias=False, featureless=args.featureless)
        
        
    print('Model initialized successfully.')


    


    features = torch.tensor(list(range(in_dim)), dtype=torch.long).to(device)
    

    model.to(device)
    
    if args.do_train:
        print("Start training...")
        train(args, features, y_train, train_mask, y_val, val_mask, y_test, test_mask, model, indice_list, weight_list)

    if args.do_valid:
        # FLAGS.dropout = 1.0
        save_dict = torch.load(os.path.join(args.save_path, 'model.bin'))
        if args.load_ckpt:
            load_ckpt(model)
        else:
            model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        # Testing
        val_cost, val_acc, pred, labels, val_duration,_ = evaluate(args,
            features, y_val, val_mask, model, indice_list, weight_list)
        print("Val set results:", "cost=", "{:.5f}".format(val_cost),
            "accuracy=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(val_duration))

        val_pred = []
        val_labels = []
        print(len(val_mask))
        for i in range(len(val_mask)):
            if val_mask[i] == 1:
                val_pred.append(pred[i])
                val_labels.append(labels[i])

        print("Val Precision, Recall and F1-Score...")
        print(metrics.classification_report(val_labels, val_pred, digits=4))
        print("Macro average Val Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(val_labels, val_pred, average='macro'))
        print("Micro average Val Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(val_labels, val_pred, average='micro'))

    if args.do_test:
        # FLAGS.dropout = 1.0
        save_dict = torch.load(os.path.join(args.save_path, 'model.bin'))
        if args.load_ckpt:
            load_ckpt(model)
        else:
            model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        # Testing
        test_cost, test_acc, pred, labels, test_duration,M = evaluate(args,
            features, y_test, test_mask, model, indice_list, weight_list)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

        test_pred = []
        test_labels = []
        print(len(test_mask))
        for i in range(len(test_mask)):
            if test_mask[i] == 1:
                test_pred.append(pred[i])
                test_labels.append(labels[i])

        print("Test Precision, Recall and F1-Score...")
        print(metrics.classification_report(test_labels, test_pred, digits=4))
        print("Macro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
        print("Micro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

        print('GP',M[0],'GR',M[1],'FS',M[2],)
if __name__ == '__main__':
    main(parse_args())
