import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torchfm.layer import MultiLayerPerceptron
import pickle
import numpy as np
from gensim.models import Word2Vec,KeyedVectors



class TGCN2(nn.Module):
    def __init__(self, in_dim, vob,hidden_dim, out_dim, num_graphs, dropout=0.1, n_layers=2, bias=False, featureless=True, act='relu'):
        
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.vob = vob
        self.num_graphs = num_graphs
        # self.bert_weight = nn.Linear(768, 300)
        self.bert_weight = nn.parameter.Parameter(torch.randn(768, 300), requires_grad=True)
        




        matirx =  self.get_embedding()

        # print('matirx:', type(matirx)   )
                # 构建三个图的初始化embedding
        # self.embedding_list = nn.ModuleList([nn.Embedding.from_pretrained(matirx, freeze=True) for _ in range(self.num_graphs)]) 
        self.embedding_list = nn.Embedding.from_pretrained(matirx, freeze=False) 


        # print('embedding_list:',self.embedding_list)

        self.layers = nn.ModuleList([GraphConv_fix(in_dim=hidden_dim, out_dim=hidden_dim, num_graphs=num_graphs, dropout=dropout, featureless=True, bias=False, act=act)])
        
        # print('layers:',self.layers)
        
        for _ in range(n_layers-1):
            self.layers.append(GraphConv_fix(in_dim=hidden_dim, out_dim=hidden_dim, num_graphs=num_graphs, dropout=dropout, featureless=False, bias=False, act=act))
        self.classifier = MultiLayerPerceptron(hidden_dim,[out_dim], dropout=dropout, output_layer=False)

        # print('hidden_dim:', hidden_dim)


    def word_dropout(self, inputs, keepprob):
        features = [self.embedding_list(inputs) for _ in range(self.num_graphs)]
        return features
    
    def get_data(self):
        with open('./data/post_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

    def get_embedding(self):
        embeddings = self.get_data()
        bert_embeddings = np.array(embeddings)
        bert_embeddings_tensor = torch.tensor(bert_embeddings)
        # bert_embeddings_tensor = nn.Embedding.from_pretrained(bert_embeddings_tensor, freeze=False) 
        bert_embeddings_tensor = bert_embeddings_tensor.float() * self.bert_weight
        # world_embedding =  KeyedVectors.load_word2vec_format('./word_embeddings/word2vec_dim300_10.kv', binary=False)
        world_embedding =  KeyedVectors.load_word2vec_format('./word_embeddings/word2vec_dim300_10.kv', binary=False)


        temp_embedding_list = torch.zeros(11533, self.hidden_dim)
        for i in range(1784):
            temp_embedding_list[i] = bert_embeddings_tensor[i]

        for i in range(len(self.vob)):
            temp_embedding_list[i+1784] = torch.tensor(world_embedding[self.vob[i]])

        for i in range(447):
            temp_embedding_list[i+11086] = bert_embeddings_tensor[i+1784]


        return temp_embedding_list




    def forward(self, inputs, edge_indexs, edge_weights, keepprob):
        # print('inputs:')
        features = self.word_dropout(inputs, keepprob)
        for layer in self.layers:
            features = layer(features, edge_indexs, edge_weights, keepprob)
        features = torch.stack(features, dim=0)
        features = torch.mean(features, dim=0)
        features = self.classifier(features)
        return features


class GraphConv_fix(nn.Module):
    def __init__(self, in_dim, out_dim, num_graphs, dropout=0.1, featureless=False, bias=False, act='relu'):
        super().__init__()
        # net_dict = {'gcn':GCNConv, 'sage':SAGEConv, 'gat':GATConv}
        # model_func = net_dict[kernel]
        self.intra_convs = nn.ModuleList([GCNConv(in_dim, out_dim,add_self_loops=False,normalize=False) for _ in range(num_graphs)])
        self.inter_convs = nn.ParameterList([nn.Parameter(torch.zeros((out_dim, out_dim), dtype=torch.float), requires_grad=True) for _ in range(num_graphs)])
        for tmp in self.inter_convs:
            nn.init.xavier_uniform_(tmp)
        if act == 'relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
            # self.act = nn.LeakyReLU()
        else:
            self.act = nn.Tanh()
        self.bias = bias
        # if self.bias:
        #     self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        #     nn.init.xavier_normal_(self.bias)
        self.dropout = nn.Dropout(dropout)
        self.featureless = featureless
        
    def atten(self, supports):
        tmp_supports = []
        for i in range(len(supports)):
            supports[i] = torch.matmul(supports[i], self.inter_convs[i])
            tmp_supports.append(supports[i])
        tmp_supports = torch.stack(tmp_supports, dim=0)
        tmp_supports_sum = torch.sum(tmp_supports, dim=0)
        att_features = []
        for support in supports:
            att_features.append(self.act(tmp_supports_sum-support))
        
        return att_features

    def forward(self, inputs, edge_indexs, edge_weights, keepprob):
        num_nodes = inputs[0].size(0)
        if not self.featureless:
            for i in range(len(inputs)):
                inputs[i] = self.dropout(inputs[i])
        supports = []
        for i, conv in enumerate(self.intra_convs):
            # support = conv(inputs[i], edge_indexs[i], edge_weights[i])
            adj = SparseTensor(row=edge_indexs[i][0], col=edge_indexs[i][1], value=edge_weights[i],
                   sparse_sizes=(num_nodes, num_nodes))
            # support = conv(inputs[i], edge_indexs[i], edge_weights[i])
            support = conv(inputs[i], adj.t())
            support = self.act(support)
            supports.append(support)

            
        supports = self.atten(supports)
        self.embedding = torch.stack(supports, dim=0)
        self.embedding = torch.mean(self.embedding, dim=0)

        return supports