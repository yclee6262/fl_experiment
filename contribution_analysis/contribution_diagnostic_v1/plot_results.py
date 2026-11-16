"""Plot all registered pair means with coverage, no fitting or new queries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root):
    summary=json.loads((root/'pair_summary.json').read_text());names=['A','anchor_L','anchor_E','full_L','full_E']
    labels=['Local sum A','Anchor / L','Anchor / E','Participant / L','Participant / E']
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.8),constrained_layout=True)
    for ax,scenario,title in [(axes[0],'clean','Clean'),(axes[1],'controlled_poison','Controlled poison')]:
        matrix=np.eye(5);coverage=np.ones((5,5))*10
        for row in summary:
            if row['scenario']!=scenario:continue
            i=names.index(row['x']);j=names.index(row['y']);matrix[i,j]=matrix[j,i]=np.nan if row['mean'] is None else row['mean'];coverage[i,j]=coverage[j,i]=row['valid_teams']
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu')
        for i in range(5):
            for j in range(5):
                if i==j:text='1.00'
                else:text=('NA' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.2f}')+f'\nn={int(coverage[i,j])}'
                ax.text(j,i,text,ha='center',va='center',fontsize=10,color='white' if abs(matrix[i,j])>.65 else '#222')
        ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_yticks(range(5),labels);ax.set_title(title)
    fig.colorbar(im,ax=axes,shrink=.65,label='Mean within-team Spearman')
    fig.suptitle('Diagnostic comparison only: 10 stored seeds per scenario; near-flat effects excluded',fontsize=12)
    path=root/'diagnostic_matrix.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root.resolve())
