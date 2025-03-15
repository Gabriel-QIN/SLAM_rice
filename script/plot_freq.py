import pandas as pd
import seqlogo
import logomaker
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO

def alntodf(aln, characters="ACDEFGHIKLMNPQRSTVWY"):
    alnRows = aln.get_alignment_length()
    compDict = {char:[0]*alnRows for char in characters}
    for record in aln:
        header = record.id
        seq = record.seq
        for aaPos in range(len(seq)):
            aa = seq[aaPos]
            if aa in characters:
                compDict[aa][aaPos] += 1    
    index = range(-25, 26)
    myd = pd.DataFrame(compDict, index=index)
    df = myd.div(myd.sum(axis=1), axis=0)
    return df

if __name__=='__main__':
    for index, ptm in enumerate(['kac', 'kcr','khib','kmal','ksucc','kla']):
        aln_f = f"../ricedata/{ptm}_train_ratio_all.fa"
        aln = AlignIO.read(aln_f, "fasta")
        fig, ax = plt.subplots(1,1, figsize=(24,5))
        df = alntodf(aln)
        # print(df)
        ww_logo = logomaker.Logo(df,
                            ax=ax,
                            color_scheme='dmslogo_funcgroup',
                            vpad=.1,
                            width=.8)
        ax.set_xlim(-25, 25)
        ticks = np.arange(-25, 26, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        # ww_logo.style_xticks(anchor=-25, spacing=5, rotation=0)
        # ww_logo.highlight_position(p=0, color='gold', alpha=.5)
        # ww_logo.highlight_position(p=26, color='gold', alpha=.5)

        # style using Axes methods
        ww_logo.ax.set_ylabel('Probability', fontsize=14, labelpad=10)
        fig.tight_layout()
        plt.savefig(f'plots/{ptm}_seqlogo.png', dpi=600)