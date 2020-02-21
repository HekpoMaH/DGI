import itertools
import numpy as np
import scipy.sparse as sp
import sklearn
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models import DGI, LogReg
from utils import process

def get_hyperparameters():
    return {
        "batch_size": 1, # Only possible setting
        "nb_epochs": 10000,
        "patience": 20,
        "lr": 0.001,
        "l2_coef": 0.0,
        "drop_prob": 0.0,
        "hid_units": 512, # 256 for larger datasets
        "nonlinearity": 'prelu', # special name to separate parameters
    }

def process_transductive(dataset):
    dataset_str = dataset
    dataset = Planetoid("./geometric_datasets"+'/'+dataset,
                        dataset,
                        transform=torch_geometric.transforms.NormalizeFeatures())[0]

    # training params
    batch_size = 1 # Transductive setting
    hyperparameters = get_hyperparameters()
    nb_epochs = hyperparameters["nb_epochs"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]
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

    mask_train = dataset.train_mask
    mask_val = dataset.val_mask
    mask_test = dataset.test_mask

    model = DGI(ft_size, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        features = features.cuda()
        labels = labels.cuda()
        edge_index = edge_index.cuda()
        mask_train = mask_train.cuda()
        mask_val = mask_val.cuda()
        mask_test = mask_test.cuda()
        model = model.cuda()


    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
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
            torch.save(model.state_dict(), 'best_dgi_'+dataset_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi_'+dataset_str+'.pkl'))
    model.eval()

    embeds, _ = model.embed(features, edge_index, None)
    train_embs = embeds[mask_train, :]
    val_embs = embeds[mask_val, :]
    test_embs = embeds[mask_test, :]

    train_lbls = labels[mask_train]
    val_lbls = labels[mask_val]
    test_lbls = labels[mask_test]

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = b_xent(logits, train_lbls)
            
            loss.backward()
            opt.step()

        logits = log(test_embs)
        print(logits.shape)
        exit(0)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())

def process_inductive(dataset):

    hyperparameters = get_hyperparameters()
    nb_epochs = hyperparameters["nb_epochs"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]
    l2_coef = hyperparameters["l2_coef"]
    drop_prob = hyperparameters["drop_prob"]
    hid_units = hyperparameters["hid_units"]
    nonlinearity = hyperparameters["nonlinearity"]
    batch_size = hyperparameters["batch_size"]

    dataset_train = PPI(
        "./geometric_datasets/"+dataset,
        split="train",
        transform=torch_geometric.transforms.NormalizeFeatures(),
    )
    print(dataset_train)
    dataset_val = PPI(
        "./geometric_datasets/"+dataset,
        split="val",
        transform=torch_geometric.transforms.NormalizeFeatures(),
    )
    print(dataset_val)
    dataset_test = PPI(
        "./geometric_datasets/"+dataset,
        split="test",
        transform=torch_geometric.transforms.NormalizeFeatures(),
    )
    print(dataset_test)

    ft_size = dataset_train[0].x.shape[1]
    nb_classes = dataset_train[0].y.shape[1] # multilabel
    model = DGI(ft_size, hid_units, nonlinearity, update_rule="MeanPool", batch_size=1)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

    loader_train = DataLoader(
        dataset_train,
        batch_size=hyperparameters["batch_size"],
        shuffle=False
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=hyperparameters["batch_size"],
        shuffle=False
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=hyperparameters["batch_size"],
        shuffle=False
    )
    model.train()

    b_xent = nn.BCEWithLogitsLoss()
    best = 1e9
    best_t = 0

    for epoch in range(20):
        break
        total_loss = 0
        batch_id = 0
        for batch in tqdm(itertools.chain(loader_train, loader_val)):
            optimiser.zero_grad()
            if torch.cuda.is_available:
                batch = batch.to('cuda')
            nb_nodes = batch.x.shape[0]
            features = batch.x
            labels = batch.y
            edge_index = batch.edge_index

            idx = np.random.randint(0, len(dataset_train))
            while idx == batch_id:
                idx = np.random.randint(0, len(dataset_train))
            shuf_fts = torch.nn.functional.dropout(dataset_train[idx].x, 0.5)
            edge_index2 = dataset_train[idx].edge_index

            lbl_1 = torch.ones(nb_nodes)
            lbl_2 = torch.zeros(shuf_fts.shape[0])
            lbl = torch.cat((lbl_1, lbl_2), 0)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                edge_index2 = edge_index2.cuda()
                lbl = lbl.cuda()
            
            logits = model(features, shuf_fts, edge_index, batch=batch.batch, edge_index_alt=edge_index2)
            # print(logits.shape, lbl.shape, lb)

            loss = b_xent(logits, lbl)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            batch_id += 1

        print(epoch, 'Loss:', total_loss/(len(dataset_train)+len(dataset_val)))

        if total_loss < best:
            best = total_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi_'+dataset+'.pkl')

    print('Loading {}th epoch'.format(best_t))
    # model.load_state_dict(torch.load('best_dgi_'+dataset+'.pkl'))
    model.eval()


    accs = []

    for _ in range(20):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        for _ in tqdm(range(100)):
            log.train()
            for l in loader_train:
                l.to('cuda')
                embeds, _ = model.embed(l.x, l.edge_index, None)
                opt.zero_grad()

                logits = log(embeds)
                loss = b_xent(logits, l.y)
                
                loss.backward()
                # print(loss.item())
                opt.step()

        tot = torch.zeros(1)
        tot = tot.cuda()
        for l in loader_test:
            l.to('cuda')
            test_embs, _ = model.embed(l.x, l.edge_index, None)
            logits = log(test_embs)
            preds = torch.sigmoid(logits) > 0.5
            # acc = torch.sum(preds == l.y).float() / l.y.shape[0]
            f1 = sklearn.metrics.f1_score(l.y.cpu(), preds.long().cpu(), average='micro')
            print(f1)
            tot += f1
        accs.append(tot/len(dataset_test))
        print('Average micro-averaged f1:', tot / len(dataset_test))

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())

dataset = "PPI"
if dataset in ("Pubmed", "Cora", "Citeseer"):
    process_transductive(dataset)
elif dataset == "PPI":
    process_inductive(dataset)
else:
    print("Unsupported dataset. Try one of {Cora, Pubmed, Citeseer} or {PPI}")
