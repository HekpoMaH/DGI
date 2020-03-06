import copy
import itertools
import numpy as np
import scipy.sparse as sp
import sklearn
import torch
import time
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as nng
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models import DGI, LogReg
from layers.GAEEnc import GAEEnc, VGAEEnc
from utils import standardise
from plot import plot_tsne


def get_hyperparameters():
    return {
        "batch_size": 1, # Only possible setting
        "nb_epochs": 10000,
        "patience": 20,
        "lr": 0.001,
        "l2_coef": 0.0000, #1*1e-3,
        "drop_prob": 0.5,
        "hid_units": 512, # 256 for larger datasets, 512 for default, 
        "nonlinearity": 'prelu', # special name to separate parameters
    }

def get_model_name(dataset_str, gnn_type, K, random_init=False, link_prediction=False, drop_sigma=False):
    assert gnn_type
    if random_init:
        prefix = 'random_'
    else:
        prefix = 'best_'
    if link_prediction:
        prefix2 = 'link_prediction_'
    else:
        prefix2 = ''
    suffix = '.pkl'

    model_name = 'dgi_'+dataset_str+'_'+gnn_type
    if K is not None and "GCNConv" not in model_name:
        model_name += '_'+str(K)
    if "SGC" in model_name and not drop_sigma:
        model_name += '_uses_sigma'
    model_name = prefix2 + prefix + model_name+suffix
    return model_name

def preprocess_embeddings(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=20,
        drop_last=False,
        shuffle=False,
        # num_workers=5
    )
    the_data = None
    for l in loader:
        the_data=l.to("cuda")

    loader = DataLoader(
        dataset,
        batch_size=2,
        drop_last=False,
        shuffle=False,
        # num_workers=5
    )
    embeds = []
    for l in loader:
        l.to("cuda")
        e, _ = model.embed(l.x, l.edge_index, None, standardise=True) 
        embeds.append(e)
    embeds = torch.cat(embeds, dim=0)

    return embeds, the_data

def train_transductive(dataset, dataset_str, edge_index, gnn_type, model_name, K=None, random_init=False, drop_sigma=False):
    batch_size = 1 # Transductive setting
    hyperparameters = get_hyperparameters()
    nb_epochs = hyperparameters["nb_epochs"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]
    if gnn_type == "SGConv":
        lr /= 3.
    hid_units = hyperparameters["hid_units"]
    nonlinearity = hyperparameters["nonlinearity"]

    nb_nodes = dataset.x.shape[0]
    ft_size = dataset.x.shape[1]
    nb_classes = torch.max(dataset.y).item()+1 # 0 based cnt
    features = dataset.x

    model = DGI(ft_size, hid_units, nonlinearity, update_rule=gnn_type, K=K)
    model_name = get_model_name(dataset_str, gnn_type, K, random_init=random_init, drop_sigma=drop_sigma)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    if torch.cuda.is_available():
        print('Using CUDA')
        features = features.cuda()
        edge_index = edge_index.cuda()
        model = model.cuda()


    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        if random_init:
            break
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]

        lbl_1 = torch.ones(nb_nodes)
        lbl_2 = torch.zeros(nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 0)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
        
        logits = model(features, shuf_fts, edge_index)

        loss = b_xent(logits, lbl)

        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), './trained_models/'+model_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()
    return best_t

def process_transductive(dataset, gnn_type='GCNConv', K=None, random_init=False, runs=20, drop_sigma=False):
    dataset_str = dataset
    norm_features = torch_geometric.transforms.NormalizeFeatures()
    dataset = Planetoid("./geometric_datasets"+'/'+dataset,
                        dataset,
                        transform=norm_features)[0]

    # training params
    batch_size = 1 # Transductive setting
    hyperparameters = get_hyperparameters()
    nb_epochs = hyperparameters["nb_epochs"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]
    xent = nn.CrossEntropyLoss()
    l2_coef = hyperparameters["l2_coef"]
    drop_prob = hyperparameters["drop_prob"]
    hid_units = hyperparameters["hid_units"]
    nonlinearity = hyperparameters["nonlinearity"]

    nb_nodes = dataset.x.shape[0]
    ft_size = dataset.x.shape[1]
    nb_classes = torch.max(dataset.y).item()+1 # 0 based cnt
    features = dataset.x
    labels = dataset.y
    edge_index = dataset.edge_index
    edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)

    mask_train = dataset.train_mask
    mask_val = dataset.val_mask
    mask_test = dataset.test_mask

    model_name = get_model_name(dataset_str, gnn_type, K, random_init=random_init)
    with open("./results/"+model_name[:-4]+"_results.txt", "w") as f:
        pass

    accs = []

    for i in range(runs): # change to how many runs you want
        model = DGI(ft_size, hid_units, nonlinearity, update_rule=gnn_type, K=K, drop_sigma=drop_sigma)
        print(model, model_name)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        if torch.cuda.is_available():
            print('Using CUDA')
            features = features.cuda()
            labels = labels.cuda()
            edge_index = edge_index.cuda()
            mask_train = mask_train.cuda()
            mask_val = mask_val.cuda()
            mask_test = mask_test.cuda()
            model = model.cuda()

        best_t = train_transductive(dataset, dataset_str, edge_index, gnn_type, model_name, K=K, random_init=random_init, drop_sigma=drop_sigma)

        xent = nn.CrossEntropyLoss()
        print('Loading {}th epoch'.format(best_t))
        print(model, model_name)
        if not random_init:
            model.load_state_dict(torch.load('./trained_models/'+model_name))
        model.eval()

        embeds, _ = model.embed(features, edge_index, None, standardise=False)
        # plot_tsne(embeds, labels, model_name)
        train_embs = embeds[mask_train, :]
        val_embs = embeds[mask_val, :]
        test_embs = embeds[mask_test, :]

        train_lbls = labels[mask_train]
        val_lbls = labels[mask_val]
        test_lbls = labels[mask_test]

        tot = torch.zeros(1)
        tot = tot.cuda()

        for _ in range(50):
            log = LogReg(hid_units, nb_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.cuda()

            pat_steps = 0
            best_acc = torch.zeros(1)
            best_acc = best_acc.cuda()
            for _ in range(150):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)
                
                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)
            print(acc)
            tot += acc

        print('Average accuracy:', tot / 50)
        
    all_accs = torch.stack(accs, dim=0)
    with open("./results/"+model_name[:-4]+"_results.txt", "a+") as f:
        f.writelines([str(all_accs.mean().item())+'\n', str(all_accs.std().item())+'\n'])

    print(all_accs.mean())
    print(all_accs.std())

def process_inductive(dataset, gnn_type="GCNConv", K=None, random_init=False):

    hyperparameters = get_hyperparameters()
    nb_epochs = hyperparameters["nb_epochs"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]
    l2_coef = hyperparameters["l2_coef"]
    drop_prob = hyperparameters["drop_prob"]
    hid_units = hyperparameters["hid_units"]
    nonlinearity = hyperparameters["nonlinearity"]
    batch_size = hyperparameters["batch_size"]

    norm_features = torch_geometric.transforms.NormalizeFeatures()
    dataset_train = PPI(
        "./geometric_datasets/"+dataset,
        split="train",
        transform=norm_features,
    )
    print(dataset_train)
    dataset_val = PPI(
        "./geometric_datasets/"+dataset,
        split="val",
        transform=norm_features,
    )
    print(dataset_val)
    dataset_test = PPI(
        "./geometric_datasets/"+dataset,
        split="test",
        transform=norm_features,
    )
    data = []
    for d in dataset_train:
        data.append(d)
    for d in dataset_val:
        data.append(d)

    ft_size = dataset_train[0].x.shape[1]
    nb_classes = dataset_train[0].y.shape[1] # multilabel

    loader_train = DataLoader(
        data,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
    )
    # loader_val = DataLoader(
    #     dataset_val,
    #     batch_size=hyperparameters["batch_size"],
    #     shuffle=False
    # )
    loader_test = DataLoader(
        dataset_test,
        batch_size=hyperparameters["batch_size"],
        shuffle=False
    )

    model = DGI(ft_size, hid_units, nonlinearity, update_rule=gnn_type, batch_size=1, K=K)
    model_name = get_model_name(dataset, gnn_type, K, random_init=random_init)
    print(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()
    model.train()

    b_xent = nn.BCEWithLogitsLoss()
    best = 1e9
    best_t = 0

    for epoch in range(20):
        if random_init:
            break
        total_loss = 0
        batch_id = 0
        model.train()
        loaded = list(loader_train)
        for batch in loaded:
            optimiser.zero_grad()
            if torch.cuda.is_available:
                batch = batch.to('cuda')
            nb_nodes = batch.x.shape[0]
            features = batch.x
            labels = batch.y
            edge_index = batch.edge_index

            idx = np.random.randint(0, len(data))
            while idx == batch_id:
                idx = np.random.randint(0, len(data))
            # idx = np.random.permutation(nb_nodes)
            # shuf_fts = features[idx, :]
            shuf_fts = torch.nn.functional.dropout(loaded[idx].x, drop_prob)
            edge_index2 = loaded[idx].edge_index

            lbl_1 = torch.ones(nb_nodes)
            lbl_2 = torch.zeros(shuf_fts.shape[0])
            lbl = torch.cat((lbl_1, lbl_2), 0)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                if edge_index2 is not None:
                    edge_index2 = edge_index2.cuda()
                lbl = lbl.cuda()
            
            logits = model(features, shuf_fts, edge_index, batch=batch.batch, edge_index_alt=edge_index2)
            # print(logits.shape, lbl.shape, lb)

            loss = b_xent(logits, lbl)
            loss.backward()
            optimiser.step()
            batch_id += 1
            total_loss += loss.item()


        print(epoch, 'Train Loss:', total_loss/(len(dataset_train)))

    torch.save(model.state_dict(), './trained_models/'+model_name)

    print('Loading last epoch')
    if not random_init:
        model.load_state_dict(torch.load('./trained_models/'+model_name))
    model.eval()
    torch.cuda.empty_cache()

    accs = []

    b_xent_reg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.25))
    train_embs, whole_train_data = preprocess_embeddings(model, dataset_train)
    val_embs, whole_val_data = preprocess_embeddings(model, dataset_val)
    test_embs, whole_test_data = preprocess_embeddings(model, dataset_test)

    for _ in range(20):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best = 1e9
        log.train()
        for _ in range(200):
            opt.zero_grad()

            logits = log(train_embs)
            loss = b_xent_reg(logits, whole_train_data.y)
            
            loss.backward()
            opt.step()

            log.eval()
            val_logits = log(val_embs) 
            loss = b_xent_reg(val_logits, whole_val_data.y)
            if loss.item() < best:
                best = loss.item()
                pat_steps = 0
            # print(loss, best, pat_steps)
            if pat_steps >= 5:
                break

            pat_steps += 1


        log.eval()
        logits = log(test_embs)
        preds = torch.sigmoid(logits) > 0.5
        # acc = torch.sum(preds == l.y).float() / l.y.shape[0]
        f1 = sklearn.metrics.f1_score(whole_test_data.y.cpu(), preds.long().cpu(), average='micro')
        accs.append(float(f1))
        print()
        print('Micro-averaged f1:', f1)

    accs = torch.tensor(accs)

    with open("./results/"+model_name[:-4]+"_results.txt", "w") as f:
        f.writelines([str(accs.mean().item())+'\n', str(accs.std().item())])
    print(accs.mean())
    print(accs.std())

def process_link_prediction(dataset, gnn_type="GCNConv", K=None):

    batch_size = 1 # Transductive setting
    hyperparameters = get_hyperparameters()
    nb_epochs = hyperparameters["nb_epochs"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]
    xent = nn.CrossEntropyLoss()
    drop_prob = hyperparameters["drop_prob"]
    hid_units = hyperparameters["hid_units"]
    nonlinearity = hyperparameters["nonlinearity"]

    dataset_str = dataset
    dataset = Planetoid("./geometric_datasets"+'/'+dataset,
                        dataset,
                        transform=torch_geometric.transforms.NormalizeFeatures())[0]
    nb_nodes = dataset.x.shape[0]
    ft_size = dataset.x.shape[1]
    nb_classes = torch.max(dataset.y).item()+1 # 0 based cnt
    features = dataset.x
    labels = dataset.y

    mask_train = dataset.train_mask
    mask_val = dataset.val_mask
    mask_test = dataset.test_mask

    model = DGI(ft_size, hid_units, nonlinearity, update_rule=gnn_type, K=K)
    model_name = get_model_name(dataset_str, gnn_type, K)
    gae = nng.GAE(model)
    gae_data = gae.split_edges(dataset)
    edge_index = gae_data.train_pos_edge_index
    print(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        print('Using CUDA')
        features = features.cuda()
        labels = labels.cuda()
        edge_index = edge_index.cuda()
        mask_train = mask_train.cuda()
        mask_val = mask_val.cuda()
        mask_test = mask_test.cuda()
        model = model.cuda()

    best_t = train_transductive(dataset, dataset_str, edge_index, gnn_type, drop_sigma=True)
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi_'+dataset_str+'_'+gnn_type+'.pkl'))
    model.eval()
    print(gae_data)
    Z = gae.encode(features, None, edge_index, embed_gae=True)
    print(gae.decode(Z, edge_index).shape)
    print(gae.test(Z, gae_data.test_pos_edge_index.cuda(), gae_data.test_neg_edge_index.cuda()))



if __name__ == "__main__":
    dataset = "Cora"
    conv = "SGConv"
    K=8
    ri = False
    LinkPrediction = False # False/True value whether we want to test link prediction
    runs=10
    drop_sigma=False
    if dataset in ("Pubmed", "Cora", "Citeseer"):
        if LinkPrediction:
            process_link_prediction(dataset, conv, K)
        else:
            for K in (2,3,4,6,8):
                process_transductive(dataset, conv, K, random_init=ri, runs=runs, drop_sigma=drop_sigma)
    elif dataset == "PPI":
        process_inductive(dataset, conv, K, random_init=ri) # conv one of {MeanPool, GATConv, SGCInductive}
    else:
        print("Unsupported dataset. Try one of {Cora, Pubmed, Citeseer} or {PPI}")
