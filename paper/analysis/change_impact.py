import json,random,pathlib,shutil,collections,sys
import numpy as np
from mtm.analysis import analyze
from mtm.validity import Run,score_groups,_tallies,_suite_scores,spearman
from itertools import combinations
base=pathlib.Path(sys.argv[1])
def rho(run,k=1,n=2000,seed=7,boot=300,bs=200):
    g=score_groups(run);names=list(g);keys={**g,'residues':run.harmful()}
    rules=[r for r in run.rule_ids if len(run.eligible[r])>=3];T=_tallies(run,rules,keys);rng=random.Random(seed);allN=names+['residues']
    def sample(rl,c):
        pools={r:list(combinations(sorted(run.eligible[r]),k)) for r in set(rl)}
        return [_suite_scores(T,rl,[rng.choice(pools[r]) for r in rl],allN) for _ in range(c)]
    def rs(rows):
        out={}
        for a in names:
            u=[r for r in rows if r[a] is not None and r['residues'] is not None]
            out[a]=spearman([r[a] for r in u],[r['residues'] for r in u])
        return out
    pt=rs(sample(rules,n));D=[];D2=[]
    for _ in range(boot):
        v=rs(sample(rng.choices(rules,k=len(rules)),bs))
        if None not in(v['rule_absent'],v['all_mutants'],v['decay_operators']):
            D.append(v['rule_absent']-v['all_mutants']);D2.append(v['all_mutants']-v['decay_operators'])
    return pt,np.percentile(D,[2.5,97.5]).round(2),np.percentile(D2,[2.5,97.5]).round(2)
def variant(name,fn):
    d=base/name;d.mkdir(parents=True,exist_ok=True);src=pathlib.Path('runs/pilot-haiku-v3')
    rows=[json.loads(l) for l in open(src/'behavior.jsonl')];rows=[x for x in (fn(dict(r)) for r in rows) if x]
    (d/'behavior.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for f in('mutants.jsonl','residues.jsonl'):shutil.copy(src/f,d/f)
    return d
def worst(r):
    if r['censored']:r['violated']=True
    return r
def v2back(r):return r
runs={'v2':pathlib.Path('runs/pilot-haiku-v2'),'v3':pathlib.Path('runs/pilot-haiku-v3'),
 'v3 censored=violation':variant('worst',worst),
 'v3 drop censored':variant('dropc',lambda r:None if r['censored'] else r),
 'v3 drop inexact resumes':variant('dropi',lambda r:None if r['resume_exact'] is False else r)}
for nm,p in runs.items():
    B=[json.loads(l) for l in open(p/'behavior.jsonl')]
    a=analyze(p);h={(x['rule_id'],x['condition']) for x in a['residues'] if x['harmful']}
    pt,d,d2=rho(Run(p))
    print(nm,'| eps',len(B),'viol',sum(b['violated'] for b in B),'R02',sum(b['violated'] for b in B if b['rule_id']=='R02'),'harmful',len(h),'detect chance',a['summary']['residue_detection_chance'],'FA',a['summary']['false_assurance_rate'],'FalseAlarm',a['summary']['false_alarm_rate'])
    print('   rho absent %.3f all %.3f decay %.3f cat %.3f llm %.3f | absent-all CI %s all-decay CI %s'%(pt['rule_absent'],pt['all_mutants'],pt['decay_operators'],pt['catastrophic_operators'],pt['llm_mutants'],d,d2))
B=[json.loads(l) for l in open('runs/pilot-haiku-v3/behavior.jsonl')]
print('censored by stratum',collections.Counter(b['condition'].split(':')[0] for b in B if b['censored']),'by rule',collections.Counter(b['rule_id'] for b in B if b['censored']))
print('inexact resumes',sum(b['resume_exact'] is False for b in B),'violated among inexact',sum(b['resume_exact'] is False and b['violated'] for b in B))
a=collections.Counter()
v2={tuple(x[k] for k in('rule_id','condition')):x['harmful'] for x in analyze(pathlib.Path('runs/pilot-haiku-v2'))['residues']}
v3={tuple(x[k] for k in('rule_id','condition')):x['harmful'] for x in analyze(pathlib.Path('runs/pilot-haiku-v3'))['residues']}
print('paired residues same',sum(v2[k]==v3[k] for k in v2),'of',len(v2),'gained',[k for k in v2 if v3[k] and not v2[k]],'lost',[k for k in v2 if v2[k] and not v3[k]])
