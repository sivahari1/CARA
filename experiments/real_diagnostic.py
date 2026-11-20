"""Seed-0 diagnostic on Notre Dame (12 test scenes) that motivated the post-hoc
CARA-lite operating point. Output: results/real_diagnostic.json"""
import sys; import os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, time
from dataclasses import replace
from cara import *
from cara import baselines as BL
from cara.real import load_notredame, subscene
from cara.method import prune_core
D=load_notredame(); rng=np.random.default_rng(500); perm=rng.permutation(715); A,Bc=perm[:357],perm[357:]
S=[subscene(D, rng.choice(A,100,replace=False), rng) for _ in range(140)]
T=[subscene(D, rng.choice(Bc,100,replace=False), rng) for _ in range(12)]
DEG=np.pi/180
OUT={}
for name,cfg in [('base',Config()),('beta5',Config(bridge_thr=5.0)),('core1',Config(core_k=1)),('beta5-core1',Config(bridge_thr=5.0,core_k=1)),('noprune-naive',Config(bridge='naive',core_k=0))]:
    t=time.time(); cal=calibrate(cfg,S[:20],S[20:60],S[60:100],S[100:140])
    a5=[];cg5=[];abst_ok=[];reg=[];cov=[]
    for sc in T:
        o=run(sc,cfg,cal); e=rot_errors(sc['R'],o['R']); ecg=rot_errors(sc['R'],BL.run_CG(sc))
        from cara.metrics import auc
        a5.append(auc(e,5)); cg5.append(auc(ecg,5)); ab=~o['reg']
        if ab.any(): abst_ok.append(np.mean(ecg[ab]<5*DEG))
        reg.append(o['reg'].mean()); r=o['reg']; cov.append(np.mean(e[r]<=o['B'][r]))
    OUT[name]=dict(auc5=float(np.mean(a5)),cg_auc5=float(np.mean(cg5)),reg=float(np.mean(reg)),cg_ok_on_abstained=float(np.mean(abst_ok)) if abst_ok else None,cov=float(np.mean(cov)))
    print(f"{name:14s} auc5 {np.mean(a5):.3f} CG {np.mean(cg5):.3f} reg {np.mean(reg):.3f} CG<5deg on abstained {np.mean(abst_ok) if abst_ok else float('nan'):.2f} cov {np.mean(cov):.3f} ({time.time()-t:.0f}s)", flush=True)

json.dump(OUT, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'real_diagnostic.json'), 'w'), indent=1)
