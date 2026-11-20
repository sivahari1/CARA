import json,os,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B=json.load(open('bounds_summary_v2.json')); S=json.load(open('summary_v2.json'))
C=["clean","fog0.7","fog0.5","fog0.3","low0.3","low0.15","rain0.5"]; L=["clean","fog\n0.7","fog\n0.5","fog\n0.3","low\n0.3","low\n0.15","rain"]
x=np.arange(len(C))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
for k,(v,c,lab) in enumerate([("base","#9aa7b8","unchanged"),("E2","#2f6db0","E2"),("E2u","#7fb0dd","E2u")]):
    ax[0].bar(x+(k-1)*0.27,[B['cov_hard_u'][cc][v] for cc in C],0.27,color=c,label=lab)
ax[0].axhline(0.9,color='k',ls='--',lw=0.8); ax[0].set_ylim(0.6,1.0); ax[0].set_xticks(x,L,fontsize=7)
ax[0].set_ylabel('coverage, hardest quarter',fontsize=8); ax[0].set_title('(a) per-camera bounds, target 0.9',fontsize=8); ax[0].legend(fontsize=6.5,ncol=3,loc='upper left',frameon=False)
for m,st in [("CG","o-"),("GLOMAP-RA","s-"),("GT-rot","k:"),("CARA","^-"),("CARA+gate","*-")]:
    ax[1].plot(x,[S['trans'][cc][m]['pos_auc'][0] for cc in C],st,ms=4,lw=1.2 if m!="CARA+gate" else 2,label={"GT-rot":"reference rot."}.get(m,m))
ax[1].set_xticks(x,L,fontsize=7); ax[1].set_ylabel('position AUC (error $\\leq 0.2$)',fontsize=8); ax[1].set_title('(b) camera positions after LUD',fontsize=8); ax[1].legend(fontsize=6.5,ncol=2,frameon=False)
for a in ax: a.tick_params(labelsize=7); a.grid(alpha=0.3,axis='y')
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper', 'figs', 'v2.pdf'))
