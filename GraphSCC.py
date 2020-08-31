from __future__ import print_function, division
import sys
import argparse
import math
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_graph, load_data_origin_data
from GNN import GCNII
from evaluation import eva
from torch.utils.data import Dataset
from collections import Counter
import h5py
import copy



from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from pre_processing import pre_processing_single
from sklearn import preprocessing
from scipy.stats import pearsonr
from calcu_graph import construct_graph_kmean
torch.set_num_threads(2)
seed=666
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        #
        self.z_layer = Linear(n_enc_3, n_z)
        #
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)


        self.x_bar_layer = Linear(n_dec_3, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar, enc_h1, enc_h2, enc_h3, z


class GraphSCC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, n_z, n_clusters, v=1):
        super(GraphSCC, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        
        self.gcn_model = GCNII(nfeat=n_input,
                      nlayers=args.nlayers,
                      nhidden=args.nhidden,
                      nclass=n_clusters,
                      dropout=0,
                      lamda=0.3,
                      alpha=0.2,
                      variant=False)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def add_noise(self, inputs):
        return inputs + (torch.randn(inputs.shape) * args.noise_value)

    def pretrain_ae(self, dataset):
        train_loader = DataLoader(dataset, batch_size=args.pre_batch_size, shuffle=True)
        optimizer = Adam(self.ae.parameters(), lr=args.pre_lr)
        for epoch in range(args.pre_epoch):
            for batch_idx, (x, _) in enumerate(train_loader):
                x_noise = self.add_noise(x)
                x_noise = x_noise.cuda()
                x = x.cuda()

                x_bar, _, _, _, z = self.ae(x_noise)
                loss = F.mse_loss(x_bar, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        predict = self.gcn_model(x, adj)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def train_graphscc(dataset):

    model = GraphSCC(512, 256, 64, 64, 256, 512,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    model.pretrain_ae(LoadDataset(dataset.x))
    optimizer = Adam(model.parameters(), lr=args.lr)
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        xbar, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=666)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    y_pred_last = y_pred

    pae_acc, pae_nmi, pae_ari = eva(y, y_pred, 'pae', pp=False)
    print(':pae_acc {:.4f}'.format(pae_acc), ', pae_nmi {:.4f}'.format(pae_nmi), ', pae_ari {:.4f}'.format(pae_ari))

    features = z.data.cpu().numpy()
    error_rate = construct_graph_kmean(args.name, features.copy(), y, y,
                                       load_type='csv', topk=args.k, method='ncos')
    adj = load_graph(args.name, k=args.k, n=dataset.x.shape[0])
    adj = adj.cuda()

    patient=0
    series=False
    for epoch in range(args.train_epoch):
        if epoch % 1 == 0:
        # update_interval
            xbar, tmp_q, pred, z = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
            Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
            P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
            print(epoch, ':Q_acc {:.5f}'.format(Q_acc), ', Q_nmi {:.5f}'.format(Q_nmi), ', Q_ari {:.5f}'.format(Q_ari))
            print(epoch, ':Z_acc {:.5f}'.format(Z_acc), ', Z_nmi {:.5f}'.format(Z_nmi), ', Z_ari {:.5f}'.format(Z_ari))
            print(epoch, ':P_acc {:.5f}'.format(P_acc), ', P_nmi {:.5f}'.format(P_nmi), ', p_ari {:.5f}'.format(p_ari))

            delta_label = np.sum(res2 != y_pred_last).astype(np.float32) / res2.shape[0]
            y_pred_last = res2
            if epoch > 0 and delta_label < 0.0001:
                if series:
                    patient+=1
                else:
                    patient = 0
                series=True
                if patient==300:
                   print('Reached tolerance threshold. Stopping training.')
                   print("Z_acc: {}".format(Z_acc), "Z_nmi: {}".format(Z_nmi),
                            "Z_ari: {}".format(Z_ari))

                   break
            else:
                series=False   


        x_bar, q, pred, _ = model(data, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = args.kl_loss * kl_loss + args.ce_loss * ce_loss + re_loss
        print(epoch, ':kl_loss {:.5f}'.format(kl_loss), ', ce_loss {:.5f}'.format(ce_loss),
              ', re_loss {:.5f}'.format(re_loss), ', total_loss {:.5f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
    Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
    P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
    print(epoch, ':Q_acc {:.4f}'.format(Q_acc), ', Q_nmi {:.4f}'.format(Q_nmi), ', Q_ari {:.4f}'.format(Q_ari))
    print(epoch, ':Z_acc {:.4f}'.format(Z_acc), ', Z_nmi {:.4f}'.format(Z_nmi), ', Z_ari {:.4f}'.format(Z_ari))
    print(epoch, ':P_acc {:.4f}'.format(P_acc), ', P_nmi {:.4f}'.format(P_nmi), ', p_ari {:.4f}'.format(p_ari))
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='klein')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--load_type', type=str, default='csv')
    parser.add_argument('--kl_loss', type=float, default=0.1)
    parser.add_argument('--ce_loss', type=float, default=0.01)
    parser.add_argument('--similar_method', type=str, default='ncos')
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--pre_epoch', type=int, default=400)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--noise_value', type=float, default=1)
    parser.add_argument('--nlayers', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    file_path = "data/" + args.name +".csv"
             
    print(args.name)
    dataset = load_data_origin_data(file_path, args.load_type,scaling=True)
    args.k = int(len(dataset.y)/100)
    if args.k < 5:
        args.k = 5
    if args.k > 20:
        args.k = 20

    args.n_clusters = len(np.unique(dataset.y))
    args.n_input = dataset.x.shape[1]
    train_graphscc(dataset)


