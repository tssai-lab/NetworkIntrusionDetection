'''

'''


import dgl.nn.pytorch as dglnn

from sklearn.metrics import confusion_matrix
import dgl


from sklearn.utils import class_weight
import dgl.nn as dglnn
import dgl.nn.pytorch
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from edgegatconv import EdgeGATConv

def compute_accuracy(pred, labels,class_num=3):
    pred= pred.argmax(1)

    acc = sklearn.metrics.accuracy_score(labels, pred)
    if class_num==2:
        micro_f1= sklearn.metrics.f1_score(labels, pred)
        macro_f1 = sklearn.metrics.f1_score(labels, pred)
    else:
        micro_f1 =  sklearn.metrics.f1_score(labels, pred,average='micro')
        macro_f1 = sklearn.metrics.f1_score(labels, pred,average='macro')


    return acc, micro_f1,macro_f1

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        #global emb
        #emb = th.cat([h_u, h_v], 1)
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Sage_Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout,class_num):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, class_num)  # 10类

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        ### force to outut fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        ### apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            # Eq4
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            # Eq5
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)



class EGAT_Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout,class_num):
        super().__init__()
        self.conv1 = EdgeGATConv(ndim_in, edim,  ndim_out, num_heads= 3  ,bias=True)
        self.conv2 = EdgeGATConv(ndim_out*3 , edim, ndim_out,num_heads=3,bias=True)
        self.relu = nn.ReLU()
        self.pred = MLPPredictor(ndim_out * 3, class_num)  # 10类

    def forward(self, g, nfeats, efeats):
        nfeats = th.squeeze(nfeats, dim=1)
        efeats = th.squeeze(efeats, dim=1)

        h = self.conv1(g,nfeats,efeats)
        h = self.relu(h.view(h.shape[0], -1))
        h = self.conv2(g,h, efeats)
        h = (h.view(h.shape[0], -1))

        return self.pred(g, h)

class GAT_Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout,class_num):
        super().__init__()
        self.conv1 = dgl.nn.pytorch.GATConv(ndim_in,   ndim_out, num_heads= 3  ,bias=True)
        self.conv2 = dgl.nn.pytorch.GATConv(ndim_out*3 ,  ndim_out,  num_heads=3,bias=True)
        self.relu = nn.ReLU()

        self.pred = MLPPredictor(ndim_out * 3, class_num)  # 10类


    def forward(self, g, nfeats, efeats):
        h = self.conv1(g,nfeats)
        h = self.relu(h.view(h.shape[0], -1))
        h = self.conv2(g,h)
        h = self.relu(h.view(h.shape[0], -1))

        return self.pred(g, h)



device = th.device('cpu' if th.cuda.is_available() else 'cpu')


data = pd.read_csv('NF-BoT-IoT.csv')

data = data.sample(frac=0.1, random_state=42)  # Random sampling of raw data

data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))


data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)


data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']

data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)

data.drop(columns=['Attack'],inplace = True)


data.rename(columns={"Label": "label"},inplace = True)


label = data.label

class_num = len(list(set(data['label']  )))
data.drop(columns=['label'],inplace = True)


scaler = StandardScaler()

data =  pd.concat([data, label], axis=1)



X_train, X_test, y_train, y_test = train_test_split(
     data, label, test_size=0.3, random_state=123,stratify= label)


encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL'])
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)


cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns ))  - set(list(['label'])) )
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])


X_train['h'] = X_train[ cols_to_norm ].values.tolist()

G = nx.from_pandas_edgelist(X_train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','label'],create_using=nx.MultiDiGraph())

#G = G.to_directed()
#G = dgl.add_self_loop(G)
G = from_networkx(G,edge_attrs=['h','label'] )


G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])

G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)
G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1,G.ndata['h'].shape[1]))
G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1,G.edata['h'].shape[1]))

# data.drop(columns=['label'],inplace = True)
# data =  pd.concat([data, label], axis=1)



#训练集的比例


print(G)



class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(G.edata['label'].cpu().numpy()),
                                                  y=G.edata['label'].cpu().numpy())

class_weights = th.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
node_features = G.ndata['h']
edge_features = G.edata['h']

edge_label = G.edata['label']
train_mask = G.edata['train_mask']


model = EGAT_Model(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, dropout= 0.2,  class_num= class_num).to(device)#

G = G.to(device)
opt = th.optim.Adam(model.parameters(),lr=0.001)
print(model)

for epoch in range(1,101):
    print('epoch: ',epoch)
    model.train()
    pred = model(G, node_features, edge_features).to(device)
    loss = criterion(pred[train_mask] ,edge_label[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    print('loss: ',loss)

    if epoch % 10 == 0:
        train_acc, train_micro_f1, train_macro_f1 = compute_accuracy(pred[train_mask].cpu(), edge_label[train_mask].cpu(), class_num)
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train micro_f1: {train_micro_f1:.4f}, Train macro_f1: {train_macro_f1:.4f}', )

'''
进行测试
'''
X_test = encoder.transform(X_test)


X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])

X_test['h'] = X_test[ cols_to_norm ].values.tolist()

G_test = nx.from_pandas_edgelist(X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','label'],create_using=nx.MultiGraph())
G_test = G_test.to_directed()
G_test = from_networkx(G_test,edge_attrs=['h','label'] )
actual = G_test.edata.pop('label')
G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G.ndata['h'].shape[2])

G_test.ndata['feature'] = th.reshape(G_test.ndata['feature'], (G_test.ndata['feature'].shape[0], 1, G_test.ndata['feature'].shape[1]))

G_test.edata['h'] = th.reshape(G_test.edata['h'], (G_test.edata['h'].shape[0], 1, G_test.edata['h'].shape[1]))



G_test = G_test.to(device)


import timeit
start_time = timeit.default_timer()
node_features_test = G_test.ndata['feature']
edge_features_test = G_test.edata['h']
test_pred = model(G_test, node_features_test, edge_features_test).to(device)
test_pred = test_pred.argmax(1)
test_pred = th.Tensor.cpu(test_pred).detach().numpy()
elapsed = timeit.default_timer() - start_time

print(str(elapsed) + ' seconds')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
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
    plt.savefig( '/home/zj/img/inductive_nfbot.png')
    plt.show()




plot_confusion_matrix(cm=confusion_matrix(edge_label, test_pred),
                      normalize=False,
                      target_names=np.unique(edge_label),
                      title="Confusion Matrix")

target_names =[ str(i) for i in list(np.unique(edge_label))]

print(classification_report(actual, test_pred, target_names=target_names, digits=4))