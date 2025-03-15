import torch_geometric
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from SLAM_seq import *
from hypara import *

matplotlib.use('agg')

threshold_data = {'seq':{"kac": {"Sp 95": 0.20, "Sp 90": 0.15, "Sp 85": 0.11, "Sp 80": 0.09},
    "kcr": {"Sp 95": 0.25, "Sp 90": 0.2, "Sp 85": 0.15, "Sp 80": 0.10},
    "khib": {"Sp 95": 0.4, "Sp 90": 0.36, "Sp 85": 0.34, "Sp 80": 0.29},
    "kmal": {"Sp 95": 0.20, "Sp 90": 0.15, "Sp 85": 0.11, "Sp 80": 0.09},
    "ksucc": {"Sp 95": 0.231041, "Sp 90": 0.228401, "Sp 85": 0.202869, "Sp 80": 0.179731},
    "kla": {"Sp 95": 0.224914, "Sp 90": 0.175495, "Sp 85": 0.142771, "Sp 80": 0.112467}}}

def parse_pdb_chain(pdb_file, chain='A',pos=None, atom_type='CA', nneighbor=32, cal_cb=True):
    """
    ########## Process PDB file ##########
    """
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    first_aa_type = None
    first_aa_position = None
    with open(pdb_file, 'r') as pdb_f:
        for line in pdb_f:
            if line[21] == chain:
                if first_aa_type is None and line[0:4].strip() == "ATOM":
                    first_aa_type = line[17:20].strip()
                    first_aa_position = int(line[22:26].strip())
                if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                    if current_aa != {}:
                        R_group = []
                        for atom in current_aa:
                            if atom not in ["N", "CA", "C", "O"]:
                                R_group.append(current_aa[atom])
                        if R_group == []:
                            R_group = [current_aa["CA"]]
                        R_group = np.array(R_group).mean(0)
                        X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                        current_aa = {}
                    if line[0:4].strip() != "TER":
                        current_pos = int(line[22:26].strip())

                if line[0:4].strip() == "ATOM":
                    atom = line[13:16].strip()
                    if atom != "H":
                        xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                        current_aa[atom] = xyz
    X = np.array(X)
    if cal_cb:
        X = np.concatenate([X, get_cb(X[:,0], X[:,1], X[:,2])[:, None]], 1)
    if pos is not None:
        atom_ind = atom_idx[atom_type] # CA atom
        if pos >= X.shape[0]:
            pos = X.shape[0] - 1
        query_coord = X[pos,atom_ind]
        distances = calculate_distances(X[:,atom_ind,:], query_coord)
        closest_indices = sorted(np.argsort(distances)[:nneighbor])
        X = X[(closest_indices)]
    return X, first_aa_type, first_aa_position  # array shape: [Length, 6, 3] N, CA, C, O, R, CB

def get_graph_fea_chain(pdb_path, pos, chain='A', nneighbor=32, radius=10, atom_type='CA', cal_cb=True):
    X, first_aa_type, first_aa_position = parse_pdb_chain(pdb_path, chain=chain,pos=pos, atom_type=atom_type, nneighbor=nneighbor, cal_cb=cal_cb)
    X = torch.tensor(X).float()
    query_atom = X[:, atom_idx[atom_type]]
    edge_index = radius_graph(query_atom, r=radius, loop=False, max_num_neighbors=nneighbor, num_workers = 4)
    node, edge = get_geo_feat(X, edge_index)
    return Data(x=node, edge_index=edge_index, edge_attr=edge, name=os.path.basename(pdb_path).split('.')[0])

def _get_encoding(seq, feature=[BLOSUM62, BINA]):
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
    # seq = [char_to_int[char] for char in sample]
    max_len = len(sample)
    all_fea = []
    for encoder in feature:
        fea = encoder([sample])
        assert fea.shape[0] == max_len
        all_fea.append(fea)
    return np.hstack(all_fea)
    
def get_all_inputs(seq, pos, tokenizer, pdb_path):
    if pdb_path is not None:
        data = get_graph_fea_chain(pdb_path, pos, nneighbor=64, atom_type='CA', cal_cb=True)
    else:
        data = None
    fea = _get_encoding(seq)
    s = ''.join([token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))])
    max_len = len(s)
    encoded = tokenizer.encode_plus(seq, add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=max_len, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return data, input_ids, attention_mask, torch.tensor(fea, dtype=torch.float)

def get_peptide(pos, window_size, seq, mirror=True):
    """Return peptide based on window_size. Missing residues are padded with X symbol (if mirror == False) or mirroring residues from the other side (if mirror == True)."""
    pos = pos-1
    half_window = int(window_size/2)
    start = pos - half_window
    left_padding = '' if start >= 0 else 'X' * abs(start)
    start = 0 if start < 0 else start
    end = pos + half_window + 1
    right_padding = 'X' * half_window
    end = len(seq) if end + 1 > len(seq) else end
    peptide_ = seq[start:end]
    if mirror:
        if left_padding == '' and right_padding == '':
            peptide = left_padding + peptide_ + right_padding
        elif left_padding == '' and right_padding != '': # mirror left
            peptide = left_padding + peptide_ + peptide_[:len(right_padding)][::-1]
        elif left_padding != '' and right_padding == '': # mirror right
            peptide = peptide_[::-1][:len(left_padding)] + peptide_ + right_padding
        else:
            peptide = None
    else:
        peptide = left_padding + peptide_ + right_padding
    if peptide is not None:
        peptide = peptide[:window_size]
        assert peptide[half_window] == 'K' and len(peptide) == window_size
        return peptide
    else:
        return None

def get_all_k(seqlist, window_size=51):
    peplist = []
    window_size = window_size
    half_window = window_size // 2
    for record in seqlist:
        name = record.id.split('|')[0]
        seq = str(record.seq)
        for m in re.finditer('K', seq):
            pos = m.start() + 1
            pep = get_peptide(pos, window_size, seq, mirror=False)
            if pep is not None:
                peplist.append([f'{name}|Pred|{pos}|{len(seq)}', pep])
    return peplist

def predict_engine(seq_path, pdb_path, threshold, ptm='kac', chain='A', use_PLM=False):
    if pdb_path is not None and use_PLM: # Use PLM+GNN
        para = HyperParam_SLAM
    else:
        if pdb_path is None: # Only use PLM
        #     para = HyperParam_struct
        # else:
            para = HyperParam_seq # Not use PLM and GNN
    gpu = para.gpu
    # device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    n_layers = para.n_layers
    dropout = para.dropout
    embedding_dim = para.embedding_dim
    hidden_dim = para.hidden_dim
    out_dim = para.out_dim
    node_dim = para.node_dim
    edge_dim = para.edge_dim
    nneighbor = para.nneighbor
    atom_type = para.atom_type
    gnn_layers = para.gnn_layers
    encoder_list = para.encoder_list.split(',')
    fea_dim = para.fea_dim
    PLM_dim = para.PLM_dim
    window_size = para.window_size
    pretrained_model = para.pretrained_model
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False)
    BERT_encoder = None
    if para.model_dir.split('/')[-1] in ['SLAM']:
        model_file = osp.join(para.model_dir, f'best_{ptm}_model_epoch.pt')
        model = SLAMNet(BERT_encoder=BERT_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,PLM_dim=PLM_dim,win_size=window_size,embedding_dim=embedding_dim, fea_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim,node_dim=node_dim, edge_dim=edge_dim, gnn_layers=gnn_layers,n_layers=n_layers,dropout=dropout).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
    else: # 'SLAM_wo_plm_and_structure'
        model_file = osp.join(para.model_dir, f'best_{ptm}_model_epoch.pt')
        model = SLAMNetSeq(BERT_encoder=None, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,win_size=window_size,embedding_dim=32, fea_dim=41, hidden_dim=64, out_dim=32, n_layers=n_layers,dropout=dropout, kernel_size=3).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
    if pdb_path is not None:
        first_aa_position = int(parse_pdb_chain(pdb_path, chain=chain,pos=None, atom_type=atom_type, nneighbor=nneighbor, cal_cb=True)[-1])
        discrepancy = first_aa_position - 1
    else:
        discrepancy = 0
    seqlist = [record for record in SeqIO.parse(seq_path, "fasta")]
    peplist = get_all_k(seqlist, window_size=window_size)

    predictions = []
    model.eval()
    for desc, seq in peplist:
        seq = str(seq)
        tmp = desc.split('|')
        pos = int(tmp[2])
        g, input_ids, attention_mask, feature = get_all_inputs(seq,pos,tokenizer,pdb_path)
        feature = feature.unsqueeze(0).to(device)
        if g is not None:
            g = g.to(device)
            g.batch = torch.zeros(g.x.shape[0],dtype=torch.int64).to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if pdb_path is not None:
            pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, g_data=g)
        else:
            pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
        score = pred.squeeze().detach().item()
        if score > threshold[0]:
            confidence = 'High'
        elif threshold[1] < score <= threshold[0]:
            confidence = 'Medium'
        elif threshold[2] <= score < threshold[1]:
            confidence = 'Low'
        else:
            confidence = f"Not modified"
        prediction = 'Yes' if score > threshold[3] else 'No'
        seq = seq
        result = [tmp[0], ptm.replace('k','K'), seq, pos+discrepancy, score, prediction, confidence]
        predictions.append(result)
        if prediction == 'Yes':
            print(result)
    
    case = pd.DataFrame(predictions)
    case.columns = ['ID', 'PTM', 'Sequence', 'Position', 'Score', 'Prediction', 'Confidence']
    kbhb = case[case['Score']>threshold[3]]
    return case, kbhb

def draw_pie(df, ptmlist=['Kac', 'Kcr', 'Khib', 'Kmal', 'Ksucc','Kla'], savepath='pie.png'):
    plt.rcParams['font.family'] = 'Times New Roman'
    # 统计不同PTM修饰在不同置信度下的占比
    ptm_confidence_counts = df.groupby(['PTM', 'Confidence']).size().unstack(fill_value=0)

    # 设置Seaborn风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # 绘制子图
    print(ptmlist)
    if len(ptmlist)  < 3:
        sub_x = 1 
        sub_y = len(ptmlist)
    elif len(ptmlist) == 3:
        sub_x = 1 
        sub_y = 3
    elif len(ptmlist) == 5:
        sub_x = 1 
        sub_y = 5
    else:
        sub_x = 2
        sub_y = len(ptmlist)//sub_x
    print(sub_x,sub_y)
    for i, ptm in enumerate(ptmlist):
        plt.subplot(sub_x, sub_y, i + 1)
        wedges, texts, autotexts = plt.pie(
            ptm_confidence_counts.loc[ptm], 
            labels=None,  # 不显示标签，避免重叠
            autopct='%1.0f%%', 
            colors=sns.color_palette("Pastel1"), 
            startangle=90,
            pctdistance=0.85  # 调整百分比显示位置
        )
        plt.title(f'{ptm}', fontsize=18)

    # 添加统一图例
    plt.figlegend(
        wedges, 
        ['High (Sp>=95%)', 'Low (Sp>=85%)', 'Medium (Sp>=90%)', 'Not modified'], 
        loc='upper right', 
        bbox_to_anchor=(1, 1), 
        fontsize=16
    )
    # plt.tight_layout()
    plt.savefig(savepath, dpi=300)

def draw_bar(df, ptmlist=['Kac', 'Kcr', 'Khib', 'Kmal', 'Ksucc','Kla'],savepath='bar.png'):
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 10))

    # 定义PTM类型
    # ptm_types = ['Kac', 'Kcr', 'Khib', 'Kmal', 'Ksucc', 'Kla']
    if len(ptmlist)  < 3:
        sub_x = 1 
        sub_y = len(ptmlist)
    elif len(ptmlist) == 3:
        sub_x = 3 
        sub_y = 1
    elif len(ptmlist) == 5:
        sub_x = 5 
        sub_y = 1
    else:
        sub_x = 2
        sub_y = len(ptmlist)//sub_x
    
    # 绘制每个PTM打分的柱状图
    for i, ptm in enumerate(ptmlist):
        plt.subplot(sub_x, sub_y, i + 1)
        sns.barplot(x='Position', y='Score', data=df[df['PTM']==ptm], palette='Pastel1',errcolor='none')
        plt.title(f'{ptm}', fontsize=16, pad=10)
        plt.xticks(rotation = 60)
    #     plt.axhline(y=threshold_data['seq'][ptm]["Sp 95"],color='red', linestyle='--', linewidth=2)
    #     plt.axhline(y=threshold_data['seq'][ptm]["Sp 90"],color='purple', linestyle='--', linewidth=2)
        plt.axhline(y=threshold_data['seq'][ptm.replace('K','k')]["Sp 80"],color='purple', linestyle='--', linewidth=2)
        plt.xlabel('Position',labelpad=10)
        plt.ylabel('Score',labelpad=10)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)

if __name__=='__main__':
    # nohup python predict_acylome.py > acylome.log 2>&1 &
    chain = 'A'
    pdb_path = None
    if pdb_path is None:
        mode = 'seq'
    else:
        mode = 'struct'
    seq_path = f'rice_proteome.fasta'
    ptm_types = 'kac' # ,kcr,khib,kmal,ksucc,kla' # 
    use_PLM = False
    pred_df = []
    all_df = []
    threshold_list = []
    for ptm in ptm_types.split(','):
        threshold = threshold_data[mode][ptm]['Sp 80'] # 
        tholds = [threshold_data[mode][ptm]['Sp 95'], threshold_data[mode][ptm]['Sp 90'], threshold_data[mode][ptm]['Sp 85'], threshold_data[mode][ptm]['Sp 80']]
        threshold_list.append(tholds)
        case, kbhb = predict_engine(seq_path, pdb_path, threshold=tholds, ptm=ptm, chain=chain, use_PLM=use_PLM)
        all_df.append(case)
        pred_df.append(kbhb)
        # print(tholds, kbhb)
    all_df = pd.concat(all_df)
    pred_df = pd.concat(pred_df)
    # print(pred_df)
    pred_df.to_csv(f'/mnt/data2024/zhqin/project/SLAM/RicePTM/Rice_{ptm_types}_high_confidence.csv', sep='\t', index=False)
    all_df.to_csv(f'/mnt/data2024/zhqin/project/SLAM/RicePTM/Rice_{ptm_types}.csv', sep='\t', index=False)
    ptm_list = [p.replace('k','K') for p in ptm_types.split(',')]
    draw_pie(all_df, ptm_list, f'/mnt/data2024/zhqin/project/SLAM/RicePTM/Rice_{ptm_types}_pie.png')
    # draw_bar(all_df, ptm_list,  'Rice_acylome_bar.png')