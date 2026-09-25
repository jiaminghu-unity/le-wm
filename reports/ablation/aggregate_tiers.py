#!/usr/bin/env python3
"""全部消融聚合(分 T1-T5):产出 a2_sr.json / a45_sr.json / a3_sr.json,
格式 {key: {"o": overall, "t": [T1..T5]}}(a45/a3 为 {"cem":..,"icem":..})。"""
import csv, glob, re, collections, json, os
S=os.path.dirname(os.path.abspath(__file__))
MP='/workspace/le-wm/eval_results/mppi_t_mirror'
WS='/workspace/le-wm/eval_results/wsweep_mirror'
TIERS=['T1','T2','T3','T4','T5']

def agg(files_seeds):  # dict seed -> list of (tier, success)
    """-> {"o":.., "t":[..]} or None"""
    suc=collections.Counter(); tot=collections.Counter()
    for rows in files_seeds.values():
        for tier,s in rows:
            suc[tier]+=s; tot[tier]+=1
    if not tot: return None
    o=100*sum(suc.values())/sum(tot.values())
    t=[round(100*suc[k]/tot[k],1) if tot[k] else None for k in TIERS]
    return {"o":round(o,1),"t":t}

def read(f):
    return [(r['tier'],int(r['success'])) for r in csv.DictReader(open(f))]

# ---------- A2 ----------
pat=re.compile(r'final_([a-z]+)_([a-z0-9_]+?)_mppiT(\d+)_s(\d+)\.csv')
A2CFG={'pusht':('c1','c3_l01'),'reacher':('r1','r2_l015'),'cube':('k1','k2'),
       'tworoom':('t1','t2_l01'),'pointmaze':('p1','p2_l01')}
need5={'102','103','104','105','106'}
d2=collections.defaultdict(dict)
for f in glob.glob(f'{MP}/final_*_mppiT*.csv'):
    m=pat.search(f); task,cfg,T,seed=m.groups()
    if task=='cube' and cfg=='k2_l01': cfg='k2'  # 同一论文模型的两种 cfg 命名
    if task not in A2CFG or cfg not in A2CFG[task] or seed not in need5: continue
    d2[(task,cfg,int(T))][seed]=read(f)
a2={}
for k,per in d2.items():
    if set(per)==need5:
        a2[f'{k[0]}|{k[1]}|{k[2]}']=agg(per)
json.dump(a2,open(f'{S}/a2_sr.json','w'))

# ---------- A4/A5 + 基线 ----------
pat2=re.compile(r'final_([a-z]+)_([a-z0-9_]+?)_(cem|icem)_s(\d+)\.csv')
M={}
for short,task in [('c3','pusht'),('r2','reacher'),('k2','cube'),('t2','tworoom'),('p2','pointmaze')]:
    for w,l in [('0.03','l003'),('0.1','l01'),('0.15','l015'),('0.3','l03'),('1.0','l1')]:
        M[f'{short}_{l}']=('obj',task,w)
for short,task in [('c5','pusht'),('r5','reacher'),('k4','cube'),('t5','tworoom'),('p5','pointmaze')]:
    for w,l in [('0.1','l01'),('0.3','l03'),('0.4','l04'),('1.0','l1')]:
        M[f'{short}_{l}']=('aux',task,w)
M.update({'k2':('obj','cube','0.1'),'t2':('obj','tworoom','0.1'),'p2':('obj','pointmaze','0.1'),
          'k4':('aux','cube','0.1'),'t5':('aux','tworoom','0.1'),'p5':('aux','pointmaze','0.1')})
for short,task in [('c1','pusht'),('r1','reacher'),('k1','cube'),('t1','tworoom'),('p1','pointmaze')]:
    M[short]=('base',task,'0')
need6={'101','102','103','104','105','106'}
d45=collections.defaultdict(dict)
for f in glob.glob(f'{WS}/final_eval*/final_*.csv'):
    m=pat2.search(f)
    if not m: continue
    task,cfg,sol,seed=m.groups()
    if cfg not in M or seed not in need6: continue
    d45[(M[cfg],sol)][seed]=read(f)
half={}
for (mk,sol),per in d45.items():
    if set(per)>=need6:
        half[(mk,sol)]=agg(per)
a45={}
for (kind,task,w) in sorted({mk for mk,_ in half}):
    c=half.get(((kind,task,w),'cem')); i=half.get(((kind,task,w),'icem'))
    if c and i: a45[f'{kind}|{task}|{w}']={"cem":c,"icem":i}
json.dump(a45,open(f'{S}/a45_sr.json','w'))

# ---------- A3 ----------
A3={'pusht|paper':'c3_l01','pusht|full':'objnat','reacher|paper':'r2_l015','reacher|full':'r2_natq',
    'cube|paper':'k2','cube|full':'qa','pointmaze|paper':'p2','pointmaze|full':'p2_natq'}
BASE={'pusht':'c1','reacher':'r1','cube':'k1','pointmaze':'p1'}
d3=collections.defaultdict(dict)
tasks={'c3_l01':'pusht','objnat':'pusht','r2_l015':'reacher','r2_natq':'reacher',
       'k2':'cube','qa':'cube','p2':'pointmaze','p2_natq':'pointmaze',
       'c1':'pusht','r1':'reacher','k1':'cube','p1':'pointmaze'}
for f in glob.glob(f'{WS}/final_eval*/final_*.csv'):
    m=pat2.search(f)
    if not m: continue
    task,cfg,sol,seed=m.groups()
    if cfg not in tasks or tasks[cfg]!=task or seed not in need6: continue
    d3[(cfg,sol)][seed]=read(f)
srt={}
for (cfg,sol),per in d3.items():
    if set(per)>=need6: srt[(cfg,sol)]=agg(per)
a3={}
for key,cfg in A3.items():
    c=srt.get((cfg,'cem')); i=srt.get((cfg,'icem'))
    if not (c and i): continue
    task=key.split('|')[0]
    bc=srt.get((BASE[task],'cem')); bi=srt.get((BASE[task],'icem'))
    a3[key]={"cem":c,"icem":i,
             "delta":f'+{round(c["o"]-bc["o"],1)} / +{round(i["o"]-bi["o"],1)}'.replace('+-','−')}
json.dump(a3,open(f'{S}/a3_sr.json','w'),ensure_ascii=False)
print('a2:',len(a2),' a45:',len(a45),' a3:',len(a3))

# ---------- 3073 复核(最优权重臂)----------
R73='/workspace/le-wm/eval_results/r73_mirror'
MAP={'c3_l03r73':'obj|pusht|0.3','c5_l1r73':'aux|pusht|1.0','k2_l1r73':'obj|cube|1.0',
     'k4_l04r73':'aux|cube|0.4','r2_l003r73':'obj|reacher|0.03','r5_l1r73':'aux|reacher|1.0',
     't2_l1r73':'obj|tworoom|1.0','t5_l1r73':'aux|tworoom|1.0','p2_l003r73':'obj|pointmaze|0.03',
     'p5_l1r73':'aux|pointmaze|1.0'}
d73=collections.defaultdict(dict)
for f in glob.glob(f'{R73}/final_eval*/final_*.csv'):
    m=pat2.search(f)
    if not m: continue
    task,cfg,sol,seed=m.groups()
    if cfg not in MAP or seed not in need6: continue
    d73[(cfg,sol)][seed]=read(f)
a73={}
for cfg,key in MAP.items():
    c=agg(d73[(cfg,'cem')]) if set(d73.get((cfg,'cem'),{}))>=need6 else None
    i=agg(d73[(cfg,'icem')]) if set(d73.get((cfg,'icem'),{}))>=need6 else None
    if c and i: a73[key]={'cem':c,'icem':i}
json.dump(a73,open(f'{S}/a73_sr.json','w'))
print('a73:',len(a73))
