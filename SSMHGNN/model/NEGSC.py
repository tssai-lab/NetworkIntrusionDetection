#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dgl.nn as dglnn
import math
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import gc
import torch.nn.init as init
from dgl.nn.functional import edge_softmax
from edgegatconv import EdgeGATConv


device = 'cpu'
file_name = 'NF-BoT-IoT.csv'
data = pd.read_csv(file_name)






data = data.groupby(by='Attack').sample(frac=0.03, random_state=13)




data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))
data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)
data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']
data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)



data.drop(columns=['Label'],inplace = True)
data.rename(columns={"Attack": "label"},inplace = True)
le = LabelEncoder()
le.fit_transform(data.label.values)
data['label'] = le.transform(data['label'])
label = data.label
data.drop(columns=['label'],inplace = True)
scaler = StandardScaler()

data =  pd.concat([data, label], axis=1)



X_train, X_test, y_train, y_test = train_test_split(
    data,label, test_size=0.3, random_state=123,stratify= label)





encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL'])

encoder.fit(X_train, y_train)

X_train = encoder.transform(X_train)





cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns ))  - set(list(['label'])) )
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])

X_train['h'] = X_train[ cols_to_norm ].values.tolist()

g_len = 35000
g_num = math.ceil(len(X_train)/g_len)

learning_rate =  0.0001
num = 20000
k1 = 3
tau = 5
num_epochs =  10000
weight_decay = 0.00001

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(39 * 6, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

        
class MultiHeadGATLayer(nn.Module):
    def __init__(self, n_feat, e_feat, out_feat, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(EdgeGATConv(n_feat, e_feat, out_feat, num_heads))
        
    def forward(self,g, h, e_feat):
        out_feat = [attn_head(g, h, e_feat) for attn_head in self.heads]
        out_feat = torch.cat(out_feat,dim = 1).reshape(g.num_nodes(),len(self.heads), -1)
        return out_feat.mean(1)

class GAT(nn.Module):
    def __init__(self, in_dim, e_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer( in_dim, e_dim, 39, num_heads)
        
    def forward(self, g, h, e_feat):
        h = self.layer1(g, h, e_feat)
        g.ndata['h'] = h
        return h , g



import random
def sub_sam(nodes, adj_lists, k):
    node_neighbor =  [ [] for i in range(nodes.shape[0])]
    node_neighbor_cen =  [ [] for i in range(nodes.shape[0])]
    node_centorr =  [[] for i in range(nodes.shape[0])]
    num_nei = 0

    for node in nodes:
        neighbors = set([int(node)])
        neighs = adj_lists[int(node)]
        node_centorr[num_nei] = [int(node)]
        current1 = adj_lists[int(node)]
        if len(neighs) >= k:
            neighs -= neighbors
            current1 = random.sample(neighs, k-1)

            node_neighbor[num_nei] = [neg_node for neg_node in current1]
            current1.append(int(node))
            node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]
            num_nei += 1

        node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]

    node_neighbor_cen = [neighbors for neighbors in node_neighbor_cen if neighbors]
    node_neighbor_cen  = node_neighbor_cen[:-1]
    return node_neighbor_cen




class Model(nn.Module):
    def __init__(self,  Encoder, gene, tau = 0.5):
        super(Model, self).__init__()
        self.encoder = Encoder
        self.tau: float = tau
        self.ge = gene

    def forward(self, graph, node_feats, edge_feats) :
        z, g1 = self.encoder(graph,node_feats,edge_feats)
        z_g, g2 = self.ge(graph,z,edge_feats)
        return z, z_g, g1, g2
    
    def embed(self, graph, node_feats, edge_feats):
        z,_ = self.encoder(graph, node_feats, edge_feats)
        return z

    def loss(self, z1, z2, adj, sub_g1, g1, g2):
        loss = self.sub_loss_batch(z1, z2, adj, sub_g1, g1, g2)
        return loss

    def sub_loss_batch(self, z, z_g, adj, sub_g1, g1, g2):
        subz_s, sub_gene_s = self.subg_centor(z, z_g, sub_g1)

        num = th.randint(0, len(sub_g1)-1, [len(sub_g1),])
        if num[0] == 0:
            num[0] = 1
        for i in range(1, len(num)):
            if num[i] == i:
                num[i] -= 1
        subg2_s_n = subz_s[num]
        sub_gene_s_n = sub_gene_s[num]
        input1 = th.cat((subz_s, subz_s, subz_s), dim=0)
        input2 = th.cat((sub_gene_s, subg2_s_n, sub_gene_s_n), dim=0)
        edges1, edges2 = self.edges_f(g1, g2, sub_g1, z, z_g)
        subg2_se = edges1[num]
        sub_gene_s_e = edges2[num]
        input1_edges = th.cat((edges1, edges1, edges1), dim=0)
        input2_edges = th.cat((edges2, subg2_se, sub_gene_s_e), dim=0)
        input1_edges = input1_edges.requires_grad_(True)
        input2_edges = input2_edges.requires_grad_(True)
        
        # adj
        subg1_adj = self.sub_adj(adj, sub_g1)
        input_adj = th.cat((subg1_adj, subg1_adj, subg1_adj), dim=0)
        
        lbl_1 = th.ones(len(sub_g1)).to(device)
        lbl_2 = th.zeros(len(sub_g1)*2).to(device)
        lbl = th.cat((lbl_1, lbl_2), 0).to(device)
        
        lbl_1_e = th.ones(len(edges2) ).to(device)
        lbl_2_e = th.zeros(len(edges2)* 2).to(device)
        lbl_e = th.cat((lbl_1_e, lbl_2_e), 0).to(device)
        
         # WD
        wd, T_wd = self.wd(input1, input2, self.tau)
        logits = th.exp(-wd / 0.01)
        loss1 = b_xent(th.squeeze(logits), lbl)
        print('loss1', loss1)
        
        # GWD
        gwd = self.gwd(input1.transpose(2,1), input2.transpose(2,1), T_wd, input_adj, self.tau)
        logits2 = th.exp(-gwd / 0.1)
        loss2 = b_xent(th.squeeze(logits2), lbl)
        print('loss2',loss2)
        

        wd, T_wd = self.wd(input1_edges, input2_edges, self.tau)
        logits3 = th.exp(-wd / 0.01)
        loss3 = b_xent(th.squeeze(logits3), lbl_e)
        print('loss3', loss3)


        loss = 0.5 * loss3 + 0.5 * loss2
        return loss
    
    def edges_f (self,g1,g2,sub_g1, z, z_g):
        edge_feat_1 = [[] for i in range(len(sub_g1))]
        edge_feat_2 = [[] for i in range(len(sub_g1))]
        sc = MLPPredictor(g1.edata['h'].shape[1],39).to(device)
        z_e = sc(g1, z)
        z_ge = sc(g2, z_g)
        for i in range(len(sub_g1)):
            cen_node = sub_g1[i][-1]
            dst = sub_g1[i][:-1]
            src_node_id = cen_node

            for j in dst:
                dst_node_id = j

                edge_indices = g1.edge_ids(src_node_id, dst_node_id,return_uv = True)

                edge_feature_1 = torch.Tensor(z_e[edge_indices[2]]).float().tolist()
                edge_feature_2 = torch.Tensor(z_ge[edge_indices[2]]).float().tolist()

                edge_feat_1.append(edge_feature_1)
                edge_feat_2.append(edge_feature_2)
                if len(edge_feat_1[-1]) == 2:
                    edge_feat_1[-1] = [edge_feat_1[-1][0]]
                    edge_feat_2[-1] = [edge_feat_2[-1][0]]
                
        edge_feat_1 = [neighbors for neighbors in edge_feat_1 if neighbors]
        edge_feat_2 = [neighbors for neighbors in edge_feat_2 if neighbors]
        edge_feat_1 = torch.Tensor(edge_feat_1)
        edge_feat_2 = torch.Tensor(edge_feat_2)
        edge_feat_1 = edge_feat_1.reshape(len(sub_g1),-1, 39)
        edge_feat_2 = edge_feat_2.reshape(len(sub_g1),-1, 39)
        return edge_feat_1, edge_feat_2
    
    def sub_adj(self, adj, sub_g1):
        subg1_adj = th.zeros(len(sub_g1), len(sub_g1[0]), len(sub_g1[0]))
        for i in range(len(sub_g1)):
            subg1_adj[i] = adj[sub_g1[i]].t()[sub_g1[i]]
        return subg1_adj


    def subg_centor(self, z, z_g, sub_g1):
        sub = [element for lis in sub_g1 for element in lis]
        subz = z[sub] 
        subg = z_g[sub]

        sub_s = subz.reshape(len(sub_g1), len(sub_g1[0]), -1)
        subg_s = subg.reshape(len(sub_g1), len(sub_g1[0]), -1)
        return sub_s, subg_s

    # WD
    def wd(self, x, y, tau):
        cos_distance = self.cost_matrix_batch(th.transpose(x, 2, 1), th.transpose(y, 2, 1), tau)
        cos_distance = cos_distance.transpose(1,2)

        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = nn.functional.relu(cos_distance - threshold)
        
        wd, T_wd = self.OT_distance_batch(cos_dist, x.size(0), x.size(1), y.size(1), 40)
        return wd, T_wd

    def OT_distance_batch(self, C, bs, n, m, iteration=50):
        C = C.float().to(device)
        T = self.OT_batch(C, bs, n, m, iteration=iteration)
        temp = th.bmm(th.transpose(C,1,2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance, T
    
    def OT_batch(self, C, bs, n, m, beta=0.5, iteration=50):
        sigma = th.ones(bs, int(m), 1).to(device)/float(m)
        T = th.ones(bs, n, m).to(device)
        A = th.exp(-C/beta).float().to(device)
        for t in range(iteration):
            Q = A * T
            for k in range(1):
                delta = 1 / (n * th.bmm(Q, sigma))
                a = th.bmm(th.transpose(Q,1,2), delta)
                sigma = 1 / (float(m) * a)
            T = delta * Q * sigma.transpose(2,1)
        return T

    def cost_matrix_batch(self, x, y, tau=0.5):
        bs = list(x.size())[0]
        D = x.size(1)
        assert(x.size(1)==y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(th.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(th.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        
        cos_dis = th.bmm(th.transpose(x, 1, 2), y)
        cos_dis = th.exp(- cos_dis / tau)
        return cos_dis.transpose(2,1)

    def batch_trace(self, input_matrix, n, bs):
        a = th.eye(n).to(device).unsqueeze(0).repeat(bs, 1, 1)
        b = a * input_matrix
        return th.sum(th.sum(b,-1),-1).unsqueeze(1)
    
    
    # GWD
    def gwd(self, X, Y, T_wd, input_adj, tau, lamda=1e-1, iteration=5, OT_iteration=20):
        m = X.size(2)
        n = Y.size(2)
        bs = X.size(0)
        p = (th.ones(bs, m, 1)/m).to(device)
        q = (th.ones(bs, n, 1)/n).to(device)
        return self.GW_distance(X, Y, p, q, T_wd, input_adj, tau, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

    def GW_distance(self, X, Y, p, q, T_wd, input_adj, tau, lamda=0.5, iteration=5, OT_iteration=20):
        cos_dis = th.exp(- input_adj / tau).to(device)
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        Cs = nn.functional.relu(res.transpose(2,1))

        Ct = self.cos_batch(Y, Y, tau).float().to(device)
        bs = Cs.size(0)
        m = Ct.size(2)
        n = Cs.size(2)
        T, Cst = self.GW_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
        temp = th.bmm(th.transpose(Cst,1,2), T_wd)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def GW_batch(self, Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
        one_m = th.ones(bs, m, 1).float().to(device)
        one_n = th.ones(bs, n, 1).float().to(device)

        Cst = th.bmm(th.bmm(Cs**2, p), th.transpose(one_m, 1, 2)) + th.bmm(one_n, th.bmm(th.transpose(q,1,2), th.transpose(Ct**2, 1, 2)))
        gamma = th.bmm(p, q.transpose(2,1))
        for i in range(iteration):
            C_gamma = Cst - 2 * th.bmm(th.bmm(Cs, gamma), th.transpose(Ct, 1, 2))
            gamma = self.OT_batch(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
        Cgamma = Cst - 2 * th.bmm(th.bmm(Cs, gamma), th.transpose(Ct, 1, 2))
        return gamma.detach(), Cgamma

    def cos_batch(self, x, y, tau):
        bs = x.size(0)
        D = x.size(1)
        assert(x.size(1)==y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(th.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(th.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = th.bmm(th.transpose(x,1,2), y)
        cos_dis = th.exp(- cos_dis / tau).transpose(1,2)
        
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        return nn.functional.relu(res.transpose(2,1))





def train(model, g,node_feats,edge_feats, adj, node_neighbor_cen):
    model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2 = model(g, node_feats, edge_feats)
    loss = model.loss(z1, z2, adj, node_neighbor_cen, g1, g2)
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss.item()



graph = []
for i in range(g_num):
    G = X_train[ i*g_len: (i+1)* g_len ]
    G = nx.from_pandas_edgelist(G, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','label'],create_using=nx.MultiGraph())
    G = G.to_directed()
    G = from_networkx(G,edge_attrs=['h','label'] )
    graph.append(G)







if graph[-1].num_nodes() < num :
    graph = graph[:-1]





n_dim = G.edata['h'].shape[1]
e_dim = G.edata['h'].shape[1]
out_dim = G.edata['h'].shape[1]
num_heads = 3
num_hidden = G.edata['h'].shape[1]
activation = F.relu



Encoder =  GAT(n_dim, e_dim, out_dim , num_heads).to(device)
gene = GAT(n_dim, e_dim, out_dim , num_heads).to(device)
model = Model(Encoder, gene).to(device)
optimizer = th.optim.Adam(model.parameters())





b_xent = nn.BCEWithLogitsLoss()
node_neighbor = {}
best = 1e9
best_t = 0
bestacc = 0




for i, g in enumerate(graph):
    g.ndata['h'] = th.ones(g.num_nodes(), g.edata['h'].shape[1])
    adj = sp.coo_matrix((np.ones(g.num_edges()), (g.edges()[0], g.edges()[1])),
                        shape=(g.num_nodes(), g.num_nodes()), dtype=np.float32).toarray()
    adj = th.from_numpy(adj).to(device)
    adj_lists = defaultdict(set)
    g1 = g
    for x in range(g1.num_edges()):
        adj_lists[g1.edges()[0][x].item()].add(g1.edges()[1][x].item())
    g = g.to(device)
    node_feats = g.ndata['h']
    edge_feats = g.edata['h']

    for epoch in range(1, num_epochs + 1):
        nodes_batch = th.randint(0, g.num_nodes(), (num,))
        node_neighbor_cen = sub_sam(nodes_batch, adj_lists, k1)
        loss = train(model, g, node_feats, edge_feats, adj, node_neighbor_cen)
    del adj, g



X_test = encoder.transform(X_test)




X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[ cols_to_norm ].values.tolist()
G_test = nx.from_pandas_edgelist(X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','label'],create_using=nx.MultiGraph())
G_test = G_test.to_directed()
G_test = from_networkx(G_test,edge_attrs=['h','label'] )
G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G_test.edata['h'].shape[1])





G_test = G_test.to(device) 




g = nx.from_pandas_edgelist(X_train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','label'],create_using=nx.MultiGraph())
g = g.to_directed()
g = from_networkx(g,edge_attrs=['h','label'] )

g.ndata['h'] = th.ones(g.num_nodes(), g.edata['h'].shape[1])




g = g.to(device)

embeds = model.embed(g,g.ndata['h'],g.edata['h']).detach()
train_embs = embeds
test_embs = model.embed(G_test,G_test.ndata['feature'],G_test.edata['h'])

train_lbls = g.edata['label']
test_lbls = G_test.edata['label']

accs = []
xent = nn.CrossEntropyLoss()


log = MLPPredictor(G.edata['h'].shape[1],len(data.label.value_counts()))
opt = th.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.0 )
log.to(device)

for _ in range(100):
    log.train()
    opt.zero_grad()

    logits = log(g, train_embs)
    loss = xent(logits, train_lbls)
    loss.backward(retain_graph=True)
    opt.step()
    print("Loss {:.4f}",loss.item())

logits = log(G_test, test_embs)
preds = th.argmax(logits, dim=1)


preds = preds.to('cpu')


test_lbls = test_lbls.to('cpu')

test_lbls = le.inverse_transform(test_lbls)
preds = le.inverse_transform(preds)




import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("confusion_matrix.png")
    plt.show()



from sklearn.metrics import confusion_matrix

plot_confusion_matrix(cm = confusion_matrix(test_lbls, preds), 
                      normalize    = True,
                      target_names = np.unique(test_lbls),
                      title        = "Confusion Matrix")





test_lbls = list(test_lbls)
preds = list(preds)




from sklearn.metrics import classification_report
target_names = np.unique(test_lbls)
print(classification_report(test_lbls, preds, target_names=target_names, digits=4))




def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    plt.show()


plot_confusion_matrix(cm = confusion_matrix(test_lbls, preds), 
                      normalize    = True,
                      target_names = np.unique(test_lbls),
                      title        = "Confusion Matrix")







