#!/bin/python
# -*- coding:utf-8 -*- 

import os
import os.path as osp
import time
import argparse
import torch
import random
import re
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm
from feature_groups import *
from metrics import eval_metrics
from collections import defaultdict
import os
import os.path as osp
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
from Bio import SeqIO

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

def BLOSUM62(fastas, **kw):
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # X
    }
    encodings = []
    for sequence in fastas:
        code = []
        for aa in sequence:
            code = blosum62[aa]
            encodings.append(code)
    arr = np.array(encodings)
    # scaler = StandardScaler().fit(arr)
    # arr = scaler.transform(arr)
    return arr

def BINA(fastas, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    encodings = []
    for sequence in fastas:
        for aa in sequence:
            if aa not in AA:
                aa = 'X'
            if aa == 'X':
                code = [0 for _ in range(len(AA))]
                encodings.append(code)
                continue
            code = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
            encodings.append(code)
    arr = np.array(encodings)
    # scaler = StandardScaler().fit(arr)
    # arr = scaler.transform(arr)
    return arr

def CC(seq):
    seq = seq[0]
    feature_CC = CC_fea().main(seq[0])
    return np.array(feature_CC)

def PCP(seq):
    seq = seq[0]
    PCP9_feature = PCP9().main(seq[0])
    return np.array(PCP9_feature)

def PWAA(seq):
    seq = seq[0]
    feature_PWAA = PWAA_fea().main(seq)
    return np.array(feature_PWAA)

def ARPC(seq):
    seq = seq[0]
    feature_ARPC = ARPC_fea().main(seq)
    return np.array(feature_ARPC)

def EBGW(seq):
    x = []
    seq = seq[0]
    feature_1_EBGW = EBGW_fea(1).main(seq)
    feature_2_EBGW = EBGW_fea(2).main(seq)
    feature_3_EBGW = EBGW_fea(3).main(seq)
    feature_4_EBGW = EBGW_fea(4).main(seq)
    feature_5_EBGW = EBGW_fea(5).main(seq)
    x.extend(feature_1_EBGW)
    x.extend(feature_2_EBGW)
    x.extend(feature_3_EBGW)
    x.extend(feature_4_EBGW)
    x.extend(feature_5_EBGW)
    return np.array(x)

def CTD(seq):
    seq = seq[0]
    feature_CTD = CTD_fea().main(seq)
    return np.array(feature_CTD)

def CKSAAP(seq):
    seq = str(seq[0])
    feature_total = []
    feature_0_space = K_space(0).main(seq)
    feature_1_space = K_space(1).main(seq)
    feature_2_space = K_space(2).main(seq)
    feature_3_space = K_space(3).main(seq)
    feature_4_space = K_space(4).main(seq)
    feature_5_space = K_space(5).main(seq)
    feature_total.extend(feature_0_space)
    feature_total.extend(feature_1_space)
    feature_total.extend(feature_2_space)
    feature_total.extend(feature_3_space)
    feature_total.extend(feature_4_space)
    feature_total.extend(feature_5_space)
    return np.array(feature_total)

class SLAMDatasetSeq(object):
    def __init__(self, seqlist, tokenizer, feature=None, fea1d=None):
        self.seq_list = []
        self.label_list = []
        self.feature_list = []
        self.fea1d_list = []
        self.tokenizer = tokenizer
        self.feature = feature
        ind = 0    
        for record in tqdm(seqlist):
            seq = str(record.seq)
            desc = record.id.split('|')
            name,label = desc[0],int(desc[1])
            if len(desc) == 3:
                pos,length = 0, 0
            else:
                pos,length = int(desc[3]),int(desc[4])
            fea = self._get_encoding(seq, feature)
            extra_fea = self.get_1d_fea(seq, fea1d)
            self.feature_list.append(fea)
            self.fea1d_list.append(extra_fea)
            self.label_list.append(int(label))
            self.seq_list.append(seq)
            ind += 1
            self.win_size = len(seq)

    def __getitem__(self, index):
        seq = self.seq_list[index]
        seq = [token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))]
        max_len = len(seq)
        encoded = self.tokenizer.encode_plus(' '.join(seq), add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].flatten()
        attention_mask = encoded['attention_mask'].flatten()
        return input_ids, attention_mask, torch.tensor(self.feature_list[index], dtype=torch.float), torch.tensor(self.fea1d_list[index], dtype=torch.float), torch.tensor(self.label_list[index], dtype=torch.long)
    
    def __len__(self):
        return len(self.seq_list)
    
    def get_1d_fea(self, seq, fea1d=[CKSAAP]):
        if fea1d == []:
            return 0.
        else:
            sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
            max_len = len(sample)
            all_fea = []
            for encoder in fea1d:
                fea = encoder([sample])
                all_fea.append(fea)
            return np.hstack(all_fea)

    def _get_encoding(self, seq, feature=[BLOSUM62, BINA]):
        sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
        max_len = len(sample)
        all_fea = []
        for encoder in feature:
            fea = encoder([sample])
            assert fea.shape[0] == max_len
            all_fea.append(fea)
        return np.hstack(all_fea)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def detach(x):
    return x.cpu().detach().numpy().squeeze()

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, win_size, out_dim=64, kernel_size=3, strides=1, dropout=0.2):
        super(CNNEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.emd = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = torch.nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        # out_channels - (kernel_size - 1) * 2 - 2
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.emd(x.long())
        x = torch.permute(x, (0,2,1))
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class PLMEncoder(nn.Module):
    def __init__(self, BERT_encoder, out_dim, kernel_size=3,PLM_dim=1024, dropout=0.2):
        super(PLMEncoder, self).__init__()
        self.bert = BERT_encoder # BertModel.from_pretrained("Rostlab/prot_bert")
        for param in self.bert.base_model.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv1d(PLM_dim, out_dim, kernel_size=kernel_size, stride=1, padding='same')
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, input_ids, attention_mask):
        pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        self.bertout = pooled_output
        imput = pooled_output.permute(0, 2, 1) # shape: (Batch, 1024, length)
        conv1_output = F.relu(self.bn1(self.conv1(imput)))  # shape: (Batch, out_channel, length)
        output = self.dropout(conv1_output)
        prot_out = torch.mean(output, axis=2, keepdim=True) # shape: (Batch, out_channel, 1)
        prot_out = prot_out.permute(0, 2, 1)  # shape: (Batch, 1, out_channel)
        return prot_out

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, n_layers, dropout, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.emd_layer = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim*2, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        if bidirectional:
            self.lstm2 = nn.LSTM(hidden_dim * 4, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.lstm2 = nn.LSTM(hidden_dim * 2, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        emd = self.emd_layer(x.long())
        self.raw_emd = emd
        output, (final_hidden_state, final_cell_state) = self.lstm1(emd.float()) # shape: (Batch, length, 128)
        output = self.dropout1(output)
        lstmout2, (_, _) = self.lstm2(output) # shape: (Batch, length, 64)
        bi_lstm_output = self.dropout2(lstmout2)
        bi_lstm_output = torch.mean(bi_lstm_output, axis=1, keepdim=True) # shape: (Batch, 1, 64)
        return bi_lstm_output

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, win_size, kernel_size=3, strides=1, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.hidden_channels = hidden_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv1 = torch.nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        # out_channels - (kernel_size - 1) * 2 - 2
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.dropout1(self.bn1(self.conv1(x.permute(0, 2, 1)))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class Fea1dEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout=0.5):
        super(Fea1dEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim//4, out_dim)
    
    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        self.fea1d = x
        logit = self.fc(x) # shape: (Batch, 1)
        return logit

class MetaDecoder(nn.Module):
    def __init__(self, combined_dim, dropout=0.5):
        super(MetaDecoder, self).__init__()
        self.fc1 = nn.Linear(combined_dim, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 5)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(5, 1)
        self.w_omega = nn.Parameter(torch.Tensor(combined_dim, combined_dim))
        self.u_omega = nn.Parameter(torch.Tensor(combined_dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))

        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        self.att_score = att_score
        scored_x = x * att_score

        context = torch.sum(scored_x, dim=1)
        return context
    
    def forward(self,fused_x):
        fusion_output = torch.cat(fused_x, axis=2) # shape: (Batch, 1, 144=64+16+64)
        self.fusion_out = fusion_output
        attn_output = self.attention_net(fusion_output)  # shape: (Batch, 80)
        self.attn_out = attn_output
        x = F.relu(self.dropout1(self.fc1(attn_output)))
        x = F.relu(self.dropout2(self.fc2(x)))
        self.final_out = x
        logit = self.fc(x) # shape: (Batch, 1)
        return logit

class SLAMNetSeq(nn.Module):
    """
    Parameter details.
    BERT_encoder: huggingface language model instance loaded by AutoModel function. e.g., 
    vocab_size: size of the dictionary of embeddings. [int]
    encoder_list: encoder used in combined model. ['cnn','lstm','plm', 'fea', 'gnn']. [list]
    PLM_dim: dimension of last hidden output for PLM.
    win_size: window size for modification-centered peptide. [int]
    embedding_dim: the size of each embedding vector.
    hidden_dim: hidden dimension for each encoder.
    out_dim: output dimension for each encoder.
    n_layers: number of BiLSTM layers.
    dropout: dropout rate.
    """
    def __init__(self, BERT_encoder, vocab_size, encoder_list=['cnn','lstm','fea'], win_size=51, embedding_dim=32, fea_dim=41, fea1d_dim=2400, hidden_dim=64, out_dim=32, kernel_size=9, bidirectional=True, n_layers=1, dropout=0.2):
        super(SLAMNetSeq, self).__init__()
        dim_list = []
        self.encoder_list = encoder_list
        if 'cnn' in self.encoder_list:
            self.cnn_encoder = CNNEncoder(vocab_size, embed_dim=embedding_dim, hidden_dim=hidden_dim, win_size=win_size, out_dim=out_dim, kernel_size=kernel_size,dropout=dropout)
            dim_list.append(out_dim)
        if 'plm' in self.encoder_list:
            self.plm_encoder = PLMEncoder(BERT_encoder=BERT_encoder, out_dim=out_dim, PLM_dim=PLM_dim, kernel_size=kernel_size, dropout=dropout)
            dim_list.append(out_dim)
        if 'lstm' in self.encoder_list:
            self.lstm_encoder = BiLSTMEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
            if bidirectional:
                dim_list.append(out_dim*2)
            else:
                dim_list.append(out_dim)
        if 'fea' in self.encoder_list:
            self.fea_encoder = FeatureEncoder(input_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim, win_size=win_size, dropout=dropout,kernel_size=kernel_size)
            dim_list.append(out_dim)
        if 'fea1d' in self.encoder_list:
            self.fea1d_encoder = Fea1dEncoder(fea1d_dim, out_dim)
            dim_list.append(out_dim)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        combined_dim = sum(dim_list)
        self.decoder = MetaDecoder(combined_dim)

    def forward(self, input_ids, attention_mask, feature=None, fea1d=None):
        fuse_x = []
        self.model_emd = {}
        if 'cnn' in self.encoder_list:
            # Encoder track 1 : CNN embeding layer
            cnn_out = self.cnn_encoder(input_ids).unsqueeze(1)
            fuse_x.append(cnn_out)
            self.model_emd['cnn_out'] = detach(cnn_out)

        if 'plm' in self.encoder_list:
            # Encoder track 2: PLM layer. shape: (Batch, length, 1024)
            prot_out = self.plm_encoder(input_ids, attention_mask)
            fuse_x.append(prot_out)
            self.model_emd['bert_out'] = detach(self.plm_encoder.bertout)
            self.model_emd['plm_out'] = detach(prot_out)

        if 'lstm' in self.encoder_list:
            # Encoder track 3 : LSTM layer.
            bi_lstm_output = self.lstm_encoder(input_ids)
            fuse_x.append(bi_lstm_output)
            self.model_emd['raw'] = detach(self.lstm_encoder.raw_emd)
            self.model_emd['lstm_out'] = detach(bi_lstm_output)

        if 'fea' in self.encoder_list and feature is not None:
            fea_out = self.fea_encoder(feature).unsqueeze(1)
            fuse_x.append(fea_out)
            self.model_emd['fea_in'] = detach(feature)
            self.model_emd['fea_out'] = detach(fea_out)
        
        if 'fea1d' in self.encoder_list and fea1d is not None:
            fea1d_out = self.fea1d_encoder(fea1d).unsqueeze(1)
            fuse_x.append(fea1d_out)
            self.model_emd['fea1d_in'] = detach(fea1d)
            self.model_emd['fea1d_out'] = detach(fea1d_out)
            
        logit = self.decoder(fuse_x)
        self.model_emd['fusion_out'] = detach(self.decoder.fusion_out)
        self.model_emd['attn_out'] = detach(self.decoder.attn_out)
        self.model_emd['final_out'] = detach(self.decoder.final_out)
        return nn.Sigmoid()(logit)

    def _extract_embedding(self):
        print(f'Extract embedding from', list(model_emd.keys()))
        return self.model_emd

def random_run(SEED=2024):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"Random seed initialization: {SEED}!")

def train_one_epoch(loader, model, device, optimizer, criterion):
    model.train()
    train_step_loss = []
    train_total_acc = 0
    step = 1
    train_total_loss = 0
    for ind,(data) in enumerate(loader):
        input_ids, attention_mask, feature, fea1d, label = data
        feature = feature.to(device)
        fea1d = fea1d.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, fea1d=fea1d)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        # print(f"Training ... Step:{step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_step_loss.append(loss.item())
        train_total_acc += acc
        step += 1
    avg_train_acc = train_total_acc / step
    avg_train_loss = train_total_loss / step
    return train_step_loss, avg_train_acc, avg_train_loss, step

def test_binary(model, loader, criterion, device):
    model.eval()
    criterion.to(device)
    test_probs = []
    test_targets = []
    valid_total_acc = 0
    valid_total_loss = 0
    valid_step = 1
    for ind,(data) in enumerate(loader):
        input_ids, attention_mask, feature, fea1d, label = data
        feature = feature.to(device)
        fea1d = fea1d.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, fea1d=fea1d)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        # print(f"Valid step:{valid_step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        valid_total_loss += loss.item()
        valid_total_acc += acc.item()
        test_probs.extend(logits.cpu().detach().numpy())
        test_targets.extend(label.cpu().detach().numpy())
        valid_step += 1

    avg_valid_loss = valid_total_loss / valid_step
    avg_valid_acc = valid_total_acc / valid_step
    # print(f"Avg Valid Loss: {avg_valid_loss:.4f} | Avg Valid Acc: {avg_valid_acc:.4f}")
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    return test_probs, test_targets, avg_valid_loss, avg_valid_acc

from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve
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
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.title('Receiver Operating Characteristic', pad=20, fontsize=16)
    for ind, name in enumerate(splist):
        targets = data[ind][0]
        probs = data[ind][1]
        fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
        plt.plot(fpr, tpr, label = f'{name}: AUROC = {auc(fpr, tpr):.4f}',linewidth=2.0)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=14, labelpad=10)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

def draw_AUPRC(data, save_path, splist=['Human', 'Mouse','False smut', 'General']):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.title('Precision Recall Curve', pad=20, fontsize=16)
    for ind, name in enumerate(splist):
        targets = data[ind][0]
        probs = data[ind][1]
        precision_1, recall_1, threshold_1 = precision_recall_curve(targets, probs)  # 计算Precision和Recall
        aupr_1 = auc(recall_1, precision_1)
        plt.plot(recall_1, precision_1, label = f'{name}: AUPRC = {aupr_1:.4f}',linewidth=2.0)
    plt.legend(loc = 'upper right')
    plt.plot([1, 0], linestyle='--', color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', fontsize=14, labelpad=10)
    plt.xlabel('Recall', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

def plot_confusion_matrix(cm, savepath, cmap=plt.cm.Blues, classes=['Non-Kbhb', 'Kbhb'], title='Confusion Matrix'):
    
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.grid(False)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='black', fontsize=15, va='center', ha='center')
            
#     norm = matplotlib.colors.Normalize(vmin=cm.min(), vmax=cm.max()*0.8)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, pad=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90, fontsize=14)
    plt.yticks(xlocations, classes, fontsize=14)
    plt.ylabel('Actual label', fontsize=16, labelpad=10)
    plt.xlabel('Predict label', fontsize=16, labelpad=10)
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
#     plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savepath, dpi=600)
def arg_parse():
    # argument parser
    parser = argparse.ArgumentParser()
    # directory and file settings
    root_dir = '/home/zhqin/project/PTM_MULTIMODAL/Datasets/nr40/win51/'
    parser.add_argument("--project_name", default='SLAM_seq', type=str,
                            help="Project name for saving model checkpoints and best model. Default:`SLAM_seq`.")
    parser.add_argument("--ptm_type", default='kac,kcr,khib,kmal,ksucc,kla', type=str,
                            help="PTM types for training.")
    parser.add_argument('--ratio', type=int, default=0, metavar='[Int]',
                        help='Negative sample ratio. (0,1,2,3 for neg:pos = all/1/2/3:1)')                    
    parser.add_argument("--train", default=osp.join(root_dir, 'general_train_ratio_1.fa'), type=str,
                            help="Data directory. Default:`'final_dataset/general_train_ratio_1.fa'`.")
    parser.add_argument("--test", default=osp.join(root_dir, 'general_test_ratio_1.fa'), type=str,
                            help="Data directory. Default:`'final_dataset/general_test_ratio_1.fa'`.")
    parser.add_argument("--model", default='Models/SLAM_seq', type=str,
                        help="Directory for model storage and logits. Default:`Models/SLAM_seq`.")
    parser.add_argument("--result", default='RicePTM_result', type=str,
                        help="Result directory for model training and evaluation. Default:`RicePTM_result`.")
    parser.add_argument("--PLM", default='/mnt/data/zhqin/pretrain_LM/prot_bert/', type=str,
                        help="PLM directory. Default:`/home/zhqin/project/pretrained_LM/prot_bert/`.")
    # Experiement settings
    parser.add_argument('--epoch', type=int, default=200, metavar='[Int]',
                        help='Number of training epochs. (default:200)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, metavar='[Float]',
                        help='Learning rate. (default:1e-4)')
    parser.add_argument('--batch', type=int, default=128, metavar='[Int]',
                        help='Batch size cutting threshold for Dataloader.(default:128)')
    parser.add_argument('--cpu', '-cpu', type=int, default=4, metavar='[Int]',
                        help='CPU processors for data loading.(default:4).')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
                        help='GPU id.(default:0).')
    parser.add_argument('--emd_dim', '-ed', type=int, default=32, metavar='[Int]',
                        help='Word embedding dimension.(default:32).')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=64, metavar='[Int]',
                        help='Hidden dimension.(default:64).')
    parser.add_argument('--out_dim', '-od', type=int, default=32, metavar='[Int]',
                        help='Out dimension for each track.(default:32).')
    parser.add_argument('--lstm_nlayer', '-ln', type=int, default=1, metavar='[Int]',
                        help='Number of LSTM layer.(default:1).') 
    parser.add_argument('--kernel_size', type=int, default=3, metavar='[Int]',
                        help='CNN kernel size.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, metavar='[Float]',
                        help='Dropout rate.(default:0.5).')                 
    parser.add_argument('--encoder', type=str, default='cnn,lstm,fea', metavar='[Str]',
                        help='Encoder list separated by comma chosen from cnn,lstm,fea,plm,gnn. (default:`cnn,lstm,fea`)')
    parser.add_argument('--fea', type=str, default=None, metavar='[Str]',
                        help='Encoder list separated by comma chosen from CKSAAP,CTD,EBGW,ARPC,PWAA,PCP,CC. (default:`None`)')
    parser.add_argument('--seed', type=int, default=2024, metavar='[Int]',
                        help='Random seed. (default:2024)')
    parser.add_argument('--patience', type=int, default=20, metavar='[Int]',
                        help='Early stopping patience. (default:20)')
    parser.add_argument('--bilstm', action="store_true", help='BiLSTM.')
    parser.add_argument('--transfer', action="store_true", help='Whether to use pretrained weights.')
    return parser.parse_args()

if __name__=='__main__':
    save_model=True
    args = arg_parse()
    print(args)
    bidirectional = True
    kernel_size = args.kernel_size
    project = args.project_name
    ptm_type = args.ptm_type.split(',')
    ratio = args.ratio
    if ratio == 0:
        ratio = 'all'
    else:
        ratio = ratio
    SEED = args.seed
    random_run(SEED)
    embedding_dim = args.emd_dim
    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    lr = args.learning_rate
    num_epochs = args.epoch
    batch_size = args.batch
    cpu = args.cpu
    gpu = args.gpu
    model_dir = osp.join(args.model, f'{project}')
    os.makedirs(model_dir, exist_ok=True)
    result_dir = osp.join(args.result, f'{project}')
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    pretrained_model = args.PLM # '/home/zhqin/project/pretrained_LM/prot_bert/'
    print(pretrained_model)
    encoder_list = args.encoder.split(',') # ['cnn','lstm','fea']
    n_layers = args.lstm_nlayer
    dropout = args.dropout
    if 'bert' in pretrained_model:
        PLM_dim = 1024
    elif 'esm' in pretrained_model:
        PLM_dim = 1280

    fea_list = ["BINA","BLOSUM62","CKSAAP","CTD","EBGW","ARPC","PWAA","PCP","CC"]
    fea_dict = {"BINA":BINA,"BLOSUM62":BLOSUM62,"CKSAAP":CKSAAP,"CTD":CTD,"EBGW":EBGW,"ARPC":ARPC,"PWAA":PWAA,"PCP":PCP,"CC":CC}
    dim_dict = {"BINA":21,"BLOSUM62":20,"CKSAAP":2400,"CTD":147,"EBGW":45,"ARPC":51,"PWAA":20,"PCP":9,"CC":18}
    manual_fea = []
    fea1d_dim = 0
    if args.fea is not None:
        for fea in args.fea.split(','):
            if fea in fea_list:
                manual_fea.append(fea_dict[fea])
                fea1d_dim += dim_dict[fea]
            else:
                print(f'{fea} is not supported currently!')
    
    if fea1d_dim > 0:
        encoder_list.append('fea1d')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False)
    if 'plm' in encoder_list:
        BERT_encoder = AutoModel.from_pretrained(pretrained_model, local_files_only=True, output_attentions=False).to(device)
    else:
        BERT_encoder = None
    fea_dim = 41 # 41
    perf_data = []
    patience_dict = {'kac':80, 'kcr':30, 'khib':80, 'kmal':10, 'ksucc': 30, 'kla':10}
    desc=['PTM', 'Ratio', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
    embedding_dict = {}
    for ptm in ptm_type:
        my_fea = [BLOSUM62, BINA] #
        train_file = f'ricedata/{ptm}_train_ratio_{ratio}.fa'
        test_file = f'ricedata/{ptm}_test_ratio_{ratio}.fa'
        train_seqlist = [record for record in SeqIO.parse(train_file, "fasta")]
        test_seqlist = [record for record in SeqIO.parse(test_file, "fasta")]
        train_ds = SLAMDatasetSeq(train_seqlist, tokenizer, feature=my_fea, fea1d=manual_fea)
        test_ds = SLAMDatasetSeq(test_seqlist, tokenizer, feature=my_fea, fea1d=manual_fea)
        window_size = test_ds.win_size
        train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=cpu,prefetch_factor=2)
        valid_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, prefetch_factor=2)
        model = SLAMNetSeq(BERT_encoder=BERT_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,win_size=window_size,embedding_dim=32, fea_dim=fea_dim, fea1d_dim=fea1d_dim, hidden_dim=64, out_dim=32, n_layers=n_layers,dropout=dropout,bidirectional=bidirectional,kernel_size=kernel_size).to(device)
        # model.apply(weight_init)
        model.load_state_dict(torch.load(f'RicePTM_result/ratio_all_no_transfer/SLAM_seq/best_{ptm}_model_epoch.pt'))
        model.eval()
        valid_step = 0
        test_probs = []
        test_targets = []
        valid_total_loss = 0
        valid_total_acc = 0
        emd_dict = defaultdict(list)
        for ind,(data) in enumerate(valid_loader):
            input_ids, attention_mask, feature, fea1d, label = data
            feature = feature.to(device)
            fea1d = fea1d.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, fea1d=fea1d)
            logits = pred.squeeze()
            # print(f"Valid step:{valid_step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
            test_probs.extend(logits.cpu().detach().numpy())
            test_targets.extend(label.cpu().detach().numpy())
            emd_dict[ind] = model.model_emd
            for key, value in model.model_emd.items():
                emd_dict[key].append(value) 
        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)
        acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(test_probs, test_targets)
        result_info = [ptm, ratio, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
        print_results(result_info, desc)
        embedding_dict[ptm] = emd_dict
        embedding_dict[ptm]['test_targets'] = test_targets
        embedding_dict[ptm]['test_logits'] = test_targets
    
    print(embedding_dict.keys())
    with open('Rice_result/embedding_no_transfer.pkl', 'wb') as pk:
        pickle.dump(embedding_dict, pk)
    
    # with open('Rice_result/embedding.pkl', 'rb') as f:
    #     embedding_dict = pickle.load(f)
    # emd_list = ['cnn_out', 'raw', 'lstm_out', 'fea_in', 'fea_out', 'fusion_out', 'attn_out', 'final_out']
    # os.makedirs(osp.join('Rice_result/plot_tsne', f'tSNE'), exist_ok=True)
    # print(osp.join('Rice_result/plot_tsne', f'tSNE'))
    # for ptm in embedding_dict.keys():
    #     emd_dict = dict(embedding_dict[ptm])
    #     test_targets = emd_dict['test_targets']
    #     for ind, (key) in enumerate(emd_list):
    #         data = emd_dict[key]
    #         data = np.concatenate(data, axis=0)
    #         if len(data.shape) > 2:
    #             data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
    #         print(f'{ptm} | {key}:\t', data.shape)
    #         save_path = osp.join('Rice_result/plot_tsne', f'tSNE/{ptm}_{key}.png')
    #         draw_tsne(test_targets, data, save_path, ptm_type=ptm)