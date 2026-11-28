import numpy as np
from mtm.rules import load_rules
from mtm import predict as P
from sklearn.metrics import roc_auc_score as A
rules={r.id:r for r in load_rules()}
ex,mu=P.load_examples('runs/pilot-haiku-v3',rules)
x,cl=P.featurize(ex,P.default_encoder())
y=np.array([e.label for e in ex],int);g=np.array([e.rule_id for e in ex]);kr=np.array([e.keyword_recall for e in ex])
lr=P.out_of_fold('logreg',x,y,list(g))
rng=np.random.default_rng(7);names=sorted(set(g));ix={r:np.flatnonzero(g==r) for r in names};D={'lr-kw':[], 'lr-sim':[], 'sim-kw':[]}
for _ in range(2000):
    i=np.concatenate([ix[names[j]] for j in rng.integers(len(names),size=len(names))])
    if len(set(y[i]))<2: continue
    a,b,c=A(y[i],lr[i]),A(y[i],-cl[i]),A(y[i],-kr[i]);D['lr-kw'].append(a-c);D['lr-sim'].append(a-b);D['sim-kw'].append(b-c)
for k,v in D.items():print(k,np.percentile(v,[2.5,97.5]).round(3))
m=kr>0
print('with some keywords: n',m.sum(),'pos',y[m].sum(),'AUROC sim',round(A(y[m],-cl[m]),3),'lr',round(A(y[m],lr[m]),3),'kw',round(A(y[m],-kr[m]),3))
