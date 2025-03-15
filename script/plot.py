#!/bin/python
# -*- coding:utf-8 -*- 

import re
import os
import os.path as osp
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import time
from Bio import SeqIO
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from SLAM_combine import *
from metrics import eval_metrics

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

def draw_tsne(test_labels, embedding, save_path, ptm_type='Kbhb'):
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for val,color,ptm in [("0", '#03BECA',f'Non-{ptm_type} sites'), ("1", '#F77672',f'{ptm_type} sites')]:
        val = int(val)
        idx = np.where(test_labels == val)
        # idx = (test_labels == val).nonzero()
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],s=4,alpha=0.6, c=color, label=ptm)
    plt.legend(loc='upper right',prop={ 'size': 10},scatterpoints=1)
    plt.savefig(save_path,dpi=600) 

# emd_list = ['raw_emd','fea_in','bert_emd','plm_out','cnn_out','lstm_out','attn_out','final_out']


def draw_performance_cm(dat, save_path, cmap=plt.cm.Purples):
    # dat = np.array([line[7] for line in result_list]).reshape(4,4)
    categories =  ['General','Human', 'Mouse','False_smut']
    # randomly generated array 
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
    # using the matshow() function  
    caxes = axes.matshow(dat, interpolation ='nearest', cmap=cmap, origin ='lower') 
    # figure.colorbar(caxes,ticks=[0, 0.2, 0.4, 0.6, 0.8, 1]) 
    # cbar = figure.colorbar(caxes)
    cmap = ccc
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = figure.colorbar(im)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    for i in range(len(dat)):
        for j in range(len(dat)):
            c = dat[i][j]
            if c > 0.001:
                plt.text(j, i, f"{c:.2f}", color='black', fontsize=10, va='center', ha='center')
                
    axes.set_xticklabels([' ']+categories, fontsize=12)
    axes.tick_params(axis='x', direction='out', pad=10)
    axes.set_yticklabels([' ']+categories, fontsize=12) 
    plt.ylabel('Models', fontsize=14, labelpad=10) # , fontweight='bold')
    plt.xlabel('Datasets', fontsize=14, labelpad=10) #, fontweight='bold')
    axes.xaxis.set_label_position('bottom') 
    axes.xaxis.tick_bottom()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


def draw_AUROC(data, save_path, splist=['Human', 'Mouse','False smut', 'General']):
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.title('Receiver Operating Characteristic', pad=20, fontsize=16)
    for ind, name in enumerate(splist):
        targets = data[ind][0]
        probs = data[ind][1]
        fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
        plt.plot(fpr, tpr, label = f'{name}: AUROC = {auc(fpr, tpr):.4f}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=14, labelpad=10)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=10)
    plt.savefig(save_path, dpi=600)

def draw_AUPRC(data, save_path, splist=['Human', 'Mouse','False smut', 'General']):
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.title('Precision Recall Curve', pad=20, fontsize=16)
    for ind, name in enumerate(splist):
        targets = data[ind][0]
        probs = data[ind][1]
        precision_1, recall_1, threshold_1 = precision_recall_curve(targets, probs)  # 计算Precision和Recall
        aupr_1 = auc(recall_1, precision_1)
        plt.plot(recall_1, precision_1, label = f'{name}: AUPRC = {aupr_1:.4f}')
    plt.legend(loc = 'lower left')
    plt.plot([1, 0], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', fontsize=14, labelpad=10)
    plt.xlabel('Recall', fontsize=14, labelpad=10)
    plt.savefig(save_path, dpi=600)

def plot_confusion_matrix(cm, savepath, cmap=plt.cm.Blues, classes=['Non-Kbhb', 'Kbhb'], title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=600)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='black', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, pad=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label', fontsize=16, labelpad=10)
    plt.xlabel('Predict label', fontsize=16, labelpad=10)
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savepath, dpi=600)


if __name__=='__main__':
    plot_dir = 'Plots/performance'
    os.makedirs(plot_dir, exist_ok=True)

    project = 'general_struct'
    root_dir = f'result/SLAM_combine/{project}'
    logits_file = osp.join(root_dir, 'logits_results.txt')
    loss_file = osp.join(root_dir, 'all_train_step_loss.txt')
    best_auc_file = osp.join(root_dir, 'best_result.csv')
    epoch_result_file = osp.join(root_dir, 'epoch_result.csv')
    best_result = pd.read_csv(best_auc_file, sep='\t')
    epoch_result = pd.read_csv(epoch_result_file, sep='\t')
    best_epoch = best_result['Epoch'].item() # epoch_result.loc[epoch_result['AUROC'].idxmax()]['Epoch'].item()
    model_file = f'Models/SLAM_combine/general_struct/best_general_struct_model_epoch79.pt'
    print(model_file)

    encoder_list = ['cnn','lstm','fea', 'gnn']
    pretrained_model = '/home/zhqin/project/pretrained_LM/prot_bert/'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False)

    if 'plm' in encoder_list:
        BERT_encoder = AutoModel.from_pretrained(pretrained_model, local_files_only=True, output_attentions=False).to(device)
    else:
        BERT_encoder = None
    if 'bert' in pretrained_model:
        PLM_dim = 1024
    elif 'esm' in pretrained_model:
        PLM_dim = 1280
    if 'fea' in encoder_list:
        manual_fea = [BLOSUM62, BINA]
        fea_dim = 41
    else:
        manual_fea=None
        fea_dim = None

    gpu = 1
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    n_layers = 1
    dropout = 0.5
    embedding_dim = 32
    hidden_dim = 64
    out_dim = 32

    node_dim = 267
    edge_dim = 632
    nneighbor = 32
    atom_type = 'CA' # CB, R, C, N, O
    gnn_layers = 5
    pdb_dir = args.pdb_dir
    test_file = '../Datasets/nr40/win51/general_test_ratio_1.fa'
    test_ds = SLAMDataset(test_file, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)
    window_size = test_ds.win_size

    test_ds = SLAMDataset(test_file, tokenizer, feature=manual_fea)
    valid_loader = DataLoader(test_ds,batch_size=128,shuffle=False,num_workers=4)
    window_size = test_ds.win_size
    model = SLAMNet(BERT_encoder=BERT_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,PLM_dim=PLM_dim,win_size=window_size,embedding_dim=embedding_dim, fea_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim,n_layers=n_layers,dropout=dropout).to(device)
    criterion = nn.BCELoss().to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    test_probs = []
    test_targets = []
    valid_total_loss = 0
    valid_total_acc = 0
    emd_dict = {}
    desc=['Project', 'Dataset', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
    start = time.perf_counter()
    for ind,(data) in enumerate(valid_loader):
        if len(data) == 4:
            samples, feature, label, g_data = data
            g_data = g_data.to(device)
        elif len(data) == 3:
            samples, feature, label = data
            feature = feature.to(device)
            g_data = None
        elif len(data) == 2:
            samples, label = data
            feature = None
            g_data = None
        input_ids = samples['input_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        print(f"Valid step:{ind} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        valid_total_loss += loss.item()
        valid_total_acc += acc.item()
        test_probs.extend(logits.cpu().detach().numpy())
        test_targets.extend(label.cpu().detach().numpy())
        emd_dict[ind] = model.model_emd
    avg_loss = valid_total_loss / len(valid_loader)
    avg_acc = valid_total_acc / len(valid_loader)
    end = time.perf_counter()
    print(f"Test | {(end - start):.4f}s | Test loss: {avg_loss:.6f}| Test acc: {avg_acc:.4f}")
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(test_probs, test_targets)
    result_info = [project, 'Test', (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
    print_results(result_info, desc)
    print(model.model_emd.keys())

    plot_project = osp.join(plot_dir, f'{project}')
    os.makedirs(plot_project, exist_ok=True)
    draw_AUROC([[test_targets, test_probs]], osp.join(plot_project, 'AUROC.png'), splist=[project])
    draw_AUPRC([[test_targets, test_probs]], osp.join(plot_project, 'AUPRC.png'), splist=[project])

    emd_list = list(model.model_emd.keys())
    os.makedirs(osp.join(plot_project, f'tSNE'), exist_ok=True)
    for emd_type in emd_list:
        data = [e[emd_type] for ind,e in emd_dict.items()]
        data = np.concatenate(data, axis=0).squeeze()
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
        print(f'{emd_type}:\t', data.shape)
        save_path = osp.join(plot_project, f'tSNE/{emd_type}.png')
        draw_tsne(test_targets, data, save_path, ptm_type='Kbhb')